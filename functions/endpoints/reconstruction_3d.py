from firebase_functions import https_fn, options
import json


@https_fn.on_request(
    timeout_sec=300,
    memory=options.MemoryOption.GB_4,
    cors=options.CorsOptions(
        cors_origins=["*"],
        cors_methods=["get", "post"],
    ))
def tridimensional_reconstruction(req: https_fn.Request) -> https_fn.Response:
    """
    Endpoint for 3D reconstruction from ultrasound mask images.
    
    Parameters (from request body):
    - transversal_image_url: URL of the transversal mask image
    - longitudinal_image_url: URL of the longitudinal mask image
    - base_T: Measure of basal thickness of transversal image (mm)
    - base_L: Measure of basal length of longitudinal image (mm)
    - height: Height (mm)
    
    Returns:
    - JSON response with GLB file URL and measurements
    """
    try:
        # Parse request body
        body_data = req.get_data().decode('utf-8').strip()
        body_json = json.loads(body_data)
        print("Received request data for 3d endpoint:", body_json)
        
        # Extract parameters
        transversal_image_url = body_json.get("transversal_image_url")
        longitudinal_image_url = body_json.get("longitudinal_image_url")
        base_T = float(body_json.get("base_T"))
        base_L = float(body_json.get("base_L"))
        height = float(body_json.get("height"))
        
        # Validate parameters
        if not transversal_image_url or not longitudinal_image_url:
            return https_fn.Response(
                response=json.dumps({"error": "Missing image URLs"}),
                status=400
            )
        
        if base_T <= 0 or base_L <= 0 or height <= 0:
            return https_fn.Response(
                response=json.dumps({"error": "All measurements must be positive"}),
                status=400
            )
        
        # Perform 3D reconstruction
        from reconstruction_3d.extrusion_reconstruction import ExtrusionReconstruction
        reconstructor = ExtrusionReconstruction()
        
        result = reconstructor.reconstruct(
            transversal_image_url=transversal_image_url,
            longitudinal_image_url=longitudinal_image_url,
            base_t_mm=base_T,
            base_l_mm=base_L,
            h_mm=height
        )
        
        if result.get("success"):
            response_data = {
                "status": "success",
                "glb_url": result["glb_url"],
                "area_mm2": result["area_mm2"],
                "volume_mm3": result["volume_mm3"],
                "processing_time": result["processing_time"]
            }
            print(response_data)
            return https_fn.Response(
                response=json.dumps(response_data),
                status=200
            )
        else:
            return https_fn.Response(
                response=json.dumps({
                    "status": "error",
                    "message": result.get("error", "Unknown error during reconstruction")
                }),
                status=500
            )
        
    except Exception as e:
        print(f"Error in tridimensional_reconstruction: {str(e)}")
        return https_fn.Response(
            response=json.dumps({
                "status": "error",
                "message": str(e)
            }),
            status=500
        )
