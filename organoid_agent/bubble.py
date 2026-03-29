"""
Simpler Bubble Generation using Convex Hull and Surface Offsetting
Much faster and fits closer to the organoid shape
"""

import numpy as np
import trimesh
import argparse
import sys
from scipy.spatial import ConvexHull


def create_offset_bubble(mesh, offset_distance=2.0):
    """
    Create a bubble by offsetting the mesh surface outward
    This fits very close to the original shape
    
    Args:
        mesh: trimesh object
        offset_distance: how far to offset (positive = expand)
    
    Returns:
        bubble_mesh: offset mesh
    """
    print(f"Creating offset bubble with distance: {offset_distance}")
    
    # Method 1: Use vertex normals to push vertices outward
    bubble = mesh.copy()
    
    # Compute vertex normals if not present
    if not hasattr(bubble, 'vertex_normals') or bubble.vertex_normals is None:
        bubble.vertex_normals = bubble.vertex_normals
    
    # Offset vertices along their normals
    bubble.vertices = bubble.vertices + bubble.vertex_normals * offset_distance
    
    # Smooth to reduce artifacts
    bubble = trimesh.smoothing.filter_laplacian(bubble, iterations=5)
    
    return bubble


def create_convex_hull_bubble(mesh):
    """
    Create a bubble using convex hull
    This creates a tight convex envelope around the shape
    
    Args:
        mesh: trimesh object
    
    Returns:
        hull_mesh: convex hull mesh
    """
    print("Creating convex hull bubble...")
    
    # Compute convex hull
    hull = mesh.convex_hull
    
    return hull


def create_alpha_shape_bubble(mesh, alpha=None):
    """
    Create a bubble using alpha shapes (concave hull)
    This fits tighter than convex hull and follows concave features
    
    Args:
        mesh: trimesh object
        alpha: alpha parameter (smaller = tighter fit, None = auto)
    
    Returns:
        alpha_mesh: alpha shape mesh
    """
    print(f"Creating alpha shape bubble (alpha={alpha})...")
    
    try:
        from scipy.spatial import Delaunay
        import alphashape
        
        # Use alphashape library if available
        alpha_shape = alphashape.alphashape(mesh.vertices, alpha)
        return alpha_shape
        
    except ImportError:
        print("alphashape not installed, falling back to convex hull")
        return create_convex_hull_bubble(mesh)


def create_subdivided_offset_bubble(mesh, offset_distance=2.0, subdivisions=2):
    """
    Create a smoother bubble by subdividing first, then offsetting
    
    Args:
        mesh: trimesh object
        offset_distance: how far to offset
        subdivisions: number of subdivision iterations
    
    Returns:
        bubble_mesh: smooth offset bubble
    """
    print(f"Creating subdivided offset bubble (offset={offset_distance}, subdivisions={subdivisions})")
    
    # Subdivide for smoother result
    bubble = mesh.copy()
    bubble = bubble.subdivide_loop(iterations=subdivisions)
    
    # Offset
    bubble.vertices = bubble.vertices + bubble.vertex_normals * offset_distance
    
    # Light smoothing
    bubble = trimesh.smoothing.filter_laplacian(bubble, iterations=3)
    
    return bubble


def create_hybrid_bubble(mesh, offset_distance=2.0, alpha=None):
    """
    Hybrid approach: alpha shape first, then slight offset
    Best balance of tight fit and smooth bubble
    
    Args:
        mesh: trimesh object
        offset_distance: additional offset after alpha shape
        alpha: alpha parameter for alpha shape
    
    Returns:
        bubble_mesh: hybrid bubble
    """
    print(f"Creating hybrid bubble (alpha={alpha}, offset={offset_distance})")
    
    # Start with alpha shape or convex hull
    try:
        import alphashape
        bubble = alphashape.alphashape(mesh.vertices, alpha)
        bubble = trimesh.Trimesh(vertices=bubble.vertices, faces=bubble.faces)
    except:
        bubble = mesh.convex_hull
    
    # Then apply small offset
    if offset_distance > 0:
        bubble.vertices = bubble.vertices + bubble.vertex_normals * offset_distance
        bubble = trimesh.smoothing.filter_laplacian(bubble, iterations=3)
    
    return bubble


def create_bubble_mesh(input_path, output_path, method='offset', offset_distance=2.0, 
                      alpha=None, subdivisions=2):
    """
    Create bubble mesh using specified method
    
    Args:
        input_path: input OBJ file
        output_path: output OBJ file
        method: 'offset', 'convex', 'alpha', 'subdivided', 'hybrid'
        offset_distance: offset distance for relevant methods
        alpha: alpha parameter for alpha shape
        subdivisions: subdivisions for subdivided method
    """
    print(f"Loading mesh from: {input_path}")
    
    # Load mesh
    try:
        mesh = trimesh.load(input_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        sys.exit(1)
    
    print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"Bounds: {mesh.bounds}")
    
    # Create bubble based on method
    if method == 'offset':
        bubble_mesh = create_offset_bubble(mesh, offset_distance)
    elif method == 'convex':
        bubble_mesh = create_convex_hull_bubble(mesh)
    elif method == 'alpha':
        bubble_mesh = create_alpha_shape_bubble(mesh, alpha)
    elif method == 'subdivided':
        bubble_mesh = create_subdivided_offset_bubble(mesh, offset_distance, subdivisions)
    elif method == 'hybrid':
        bubble_mesh = create_hybrid_bubble(mesh, offset_distance, alpha)
    else:
        print(f"Unknown method: {method}")
        sys.exit(1)
    
    print(f"Bubble mesh: {len(bubble_mesh.vertices)} vertices, {len(bubble_mesh.faces)} faces")
    
    # Save bubble
    print(f"Saving bubble mesh to: {output_path}")
    bubble_mesh.export(output_path)
    
    # Save combined visualization
    combined_path = output_path.replace('.obj', '_combined.obj')
    
    original_colored = mesh.copy()
    original_colored.visual.vertex_colors = [255, 100, 100, 200]  # Red, semi-transparent
    
    bubble_colored = bubble_mesh.copy()
    bubble_colored.visual.vertex_colors = [100, 150, 255, 120]  # Blue, semi-transparent
    
    combined = trimesh.util.concatenate([original_colored, bubble_colored])
    combined.export(combined_path)
    print(f"Saved combined visualization to: {combined_path}")
    
    # Calculate distance statistics
    print("\nAnalyzing bubble fit...")
    distances = trimesh.proximity.signed_distance(bubble_mesh, mesh.vertices)
    print(f"Distance statistics:")
    print(f"  Mean: {np.mean(distances):.3f}")
    print(f"  Std: {np.std(distances):.3f}")
    print(f"  Min: {np.min(distances):.3f}")
    print(f"  Max: {np.max(distances):.3f}")
    
    return bubble_mesh, mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create bubble around organoid mesh (simpler methods)"
    )
    parser.add_argument("input", help="Input OBJ file path")
    parser.add_argument("-o", "--output", default="bubble_output.obj",
                       help="Output OBJ file path")
    parser.add_argument("-m", "--method", 
                       choices=['offset', 'convex', 'alpha', 'subdivided', 'hybrid'],
                       default='offset',
                       help="Bubble creation method (default: offset)")
    parser.add_argument("-d", "--distance", type=float, default=0.5,
                       help="Offset distance (default: 2.0)")
    parser.add_argument("-a", "--alpha", type=float, default=None,
                       help="Alpha parameter for alpha shape (default: auto)")
    parser.add_argument("-s", "--subdivisions", type=int, default=2,
                       help="Subdivisions for subdivided method (default: 2)")
    
    args = parser.parse_args()
    
    bubble_mesh, original_mesh = create_bubble_mesh(
        args.input,
        args.output,
        method=args.method,
        offset_distance=args.distance,
        alpha=args.alpha,
        subdivisions=args.subdivisions
    )
    
    print("\n✓ Complete!")
    print(f"  Bubble mesh: {args.output}")
    print(f"  Combined visualization: {args.output.replace('.obj', '_combined.obj')}")
    print(f"\nRecommended methods:")
    print(f"  - For closest fit: python script.py input.obj -m offset -d 1.0")
    print(f"  - For smooth bubble: python script.py input.obj -m subdivided -d 2.0")
    print(f"  - For convex envelope: python script.py input.obj -m convex")
    print(f"  - For best balance: python script.py input.obj -m hybrid -d 1.5")
