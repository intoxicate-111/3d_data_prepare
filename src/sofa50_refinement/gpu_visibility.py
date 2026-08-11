from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def compute_renderer_visibility_cuda(
    mesh: Any,
    cameras: Sequence[Any],
    *,
    image_size: int,
    neighborhood_radius: int = 1,
    front_face_winding: str = "ccw",
) -> Any:
    """Compute the downstream renderer-visibility contract with nvdiffrast.

    Face-ID rasterization runs on CUDA. Frustum, projected winding and the
    incident-face neighborhood test retain the downstream definitions.
    """

    if neighborhood_radius < 0:
        raise ValueError("neighborhood_radius must be non-negative")
    if front_face_winding not in {"ccw", "cw"}:
        raise ValueError("front_face_winding must be ccw or cw")
    if image_size < 1:
        raise ValueError("image_size must be positive")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA visibility requires torch.cuda.is_available()")
    try:
        import nvdiffrast.torch as dr
    except ImportError as exc:
        raise RuntimeError(
            "CUDA visibility requires nvdiffrast on PYTHONPATH."
        ) from exc

    from mlr.learned_laplacian.renderer_visibility import (
        RendererVisibilityResult,
        frustum_valid,
        projected_backface_visibility_with_counts,
        vertex_visibility_from_face_id_buffer,
    )

    mesh.ensure_normals()
    vertices_np = np.asarray(mesh.vertices, dtype=np.float32)
    faces_np = np.asarray(mesh.faces, dtype=np.int64)
    device = torch.device("cuda")
    vertices = torch.as_tensor(vertices_np, dtype=torch.float32, device=device)
    faces = torch.as_tensor(faces_np, dtype=torch.long, device=device)
    context = dr.RasterizeCudaContext(device=device)

    num_views = len(cameras)
    num_vertices = mesh.num_vertices
    frustum = np.zeros((num_views, num_vertices), dtype=bool)
    backface = np.zeros_like(frustum)
    occlusion = np.zeros_like(frustum)
    combined = np.zeros_like(frustum)
    front_counts = np.zeros(num_views, dtype=np.int64)
    back_counts = np.zeros(num_views, dtype=np.int64)
    two_sided_pixels = np.zeros(num_views, dtype=np.int64)
    culled_pixels = np.zeros(num_views, dtype=np.int64)

    for view_index, camera in enumerate(cameras):
        if camera.image_size is not None and tuple(camera.image_size) != (
            image_size,
            image_size,
        ):
            raise ValueError(
                f"camera {view_index} image size {camera.image_size} differs from "
                f"{image_size}x{image_size}"
            )
        frustum[view_index] = frustum_valid(mesh.vertices, camera)
        (
            backface[view_index],
            front_counts[view_index],
            back_counts[view_index],
        ) = projected_backface_visibility_with_counts(
            mesh, camera, front_face_winding
        )
        clip, pixels, positive = _camera_clip_positions(
            vertices, faces, camera, image_size
        )
        area2 = _triangle_area2(pixels[faces])
        nondegenerate = torch.abs(area2) > 1e-12
        two_sided_faces = torch.nonzero(
            positive & nondegenerate, as_tuple=False
        ).flatten()
        front = area2 < 0.0 if front_face_winding == "ccw" else area2 > 0.0
        culled_faces = torch.nonzero(
            positive & nondegenerate & front, as_tuple=False
        ).flatten()

        two_sided_ids = _rasterize_face_ids(
            context,
            dr,
            clip,
            faces,
            two_sided_faces,
            image_size,
        )
        culled_ids = _rasterize_face_ids(
            context,
            dr,
            clip,
            faces,
            culled_faces,
            image_size,
        )
        two_sided_pixels[view_index] = int(np.count_nonzero(two_sided_ids >= 0))
        culled_pixels[view_index] = int(np.count_nonzero(culled_ids >= 0))
        occlusion[view_index] = vertex_visibility_from_face_id_buffer(
            mesh.vertices,
            mesh.faces,
            camera,
            two_sided_ids,
            neighborhood_radius=neighborhood_radius,
        )
        combined[view_index] = vertex_visibility_from_face_id_buffer(
            mesh.vertices,
            mesh.faces,
            camera,
            culled_ids,
            neighborhood_radius=neighborhood_radius,
        )

    return RendererVisibilityResult(
        frustum_valid=frustum,
        backface_visible=backface,
        occlusion_visible=occlusion,
        backface_and_occlusion_visible=combined,
        neighborhood_radius=int(neighborhood_radius),
        backend="cuda_nvdiffrast",
        front_face_winding=front_face_winding,
        front_face_counts=front_counts,
        back_face_counts=back_counts,
        two_sided_pixel_counts=two_sided_pixels,
        culled_pixel_counts=culled_pixels,
    )


def _camera_clip_positions(vertices: Any, faces: Any, camera: Any, image_size: int):
    import torch

    rotation = torch.as_tensor(
        camera.rotation, dtype=torch.float32, device=vertices.device
    )
    translation = torch.as_tensor(
        camera.translation, dtype=torch.float32, device=vertices.device
    )
    intrinsics = torch.as_tensor(
        camera.intrinsics, dtype=torch.float32, device=vertices.device
    )
    camera_vertices = vertices @ rotation.T + translation.unsqueeze(0)
    z = camera_vertices[:, 2]
    safe_z = torch.where(torch.abs(z) > 1e-12, z, torch.ones_like(z))
    projected_h = camera_vertices @ intrinsics.T
    pixels = projected_h[:, :2] / safe_z.unsqueeze(1)

    # Pixel x points right and pixel y points down. nvdiffrast clip y points up.
    # Multiplication by z supplies the perspective divide expected by clip space.
    clip_x = (2.0 * pixels[:, 0] / float(image_size) - 1.0) * z
    clip_y = (1.0 - 2.0 * pixels[:, 1] / float(image_size)) * z
    near = torch.clamp(z[z > 1e-8].amin() * 0.5, min=1e-6)
    far = torch.clamp(z[z > 1e-8].amax() * 2.0, min=near + 1.0)
    a = (far + near) / (far - near)
    b = -2.0 * far * near / (far - near)
    clip_z = a * z + b
    clip = torch.stack((clip_x, clip_y, clip_z, z), dim=1).contiguous()
    positive = torch.all(z[faces] > 1e-8, dim=1)
    return clip, pixels, positive


def _triangle_area2(triangle_pixels: Any):
    edge_a = triangle_pixels[:, 1] - triangle_pixels[:, 0]
    edge_b = triangle_pixels[:, 2] - triangle_pixels[:, 0]
    return edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]


def _rasterize_face_ids(
    context: Any,
    dr: Any,
    clip: Any,
    faces: Any,
    selected_face_ids: Any,
    image_size: int,
) -> np.ndarray:
    import torch

    if selected_face_ids.numel() == 0:
        return np.full((image_size, image_size), -1, dtype=np.int32)
    selected_faces = faces[selected_face_ids].to(dtype=torch.int32).contiguous()
    rast, _ = dr.rasterize(
        context,
        clip.unsqueeze(0),
        selected_faces,
        resolution=[image_size, image_size],
    )
    local_ids = rast[0, :, :, 3].to(dtype=torch.long) - 1
    valid = local_ids >= 0
    original_ids = torch.full_like(local_ids, -1)
    original_ids[valid] = selected_face_ids[local_ids[valid]]
    # nvdiffrast returns scanlines from bottom to top; downstream arrays index
    # rows in CV pixel order from top to bottom.
    return torch.flip(original_ids, dims=(0,)).to(dtype=torch.int32).cpu().numpy()
