import requests
import time
import base64

# Your Meshy AI API key
API_KEY = "msy_IUda5lvaIW42LnFnU6R7pN8ZbJUST3IhDBYp"

# Base URL
BASE_URL = "https://api.meshy.ai/v1/image-to-3d"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Step 1: Create the 3D generation task
def create_image_to_3d_task(image_path, enable_pbr=False):
    """
    Create an image-to-3D task
    
    Args:
        image_path: Path to your image file
        enable_pbr: Whether to generate PBR materials
    """
    # Read and encode image as base64
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "image_url": f"data:image/jpeg;base64,{image_data}",  # or jpeg
        "ai_model": "latest",
        "topology": "quad",
        "symmetry_mode": "on", 
        "should_remesh": True,
        "should_texture": False,
        "save_pre_remeshed_model": False,
        "enable_pbr": False,        
    }
    
    response = requests.post(BASE_URL, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    task_id = result['result']
    print(f"Task created with ID: {task_id}")
    return task_id

# Step 2: Check task status
def get_task_status(task_id):
    """Check the status of a 3D generation task"""
    url = f"{BASE_URL}/{task_id}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

# Step 3: Poll until complete and download
def wait_and_download(task_id, output_prefix="output"):
    """
    Wait for task to complete and download results
    """
    print("Waiting for 3D generation to complete...")
    
    while True:
        status_data = get_task_status(task_id)
        status = status_data['status']
        
        print(f"Status: {status}")
        
        if status == "SUCCEEDED":
            print("Generation complete!")
            
            # Download model files
            model_urls = status_data.get('model_urls', {})
            
            # Download GLB
            if 'glb' in model_urls:
                glb_url = model_urls['glb']
                glb_response = requests.get(glb_url)
                with open(f"{output_prefix}.glb", 'wb') as f:
                    f.write(glb_response.content)
                print(f"Downloaded: {output_prefix}.glb")
            
            # Download FBX
            if 'fbx' in model_urls:
                fbx_url = model_urls['fbx']
                fbx_response = requests.get(fbx_url)
                with open(f"{output_prefix}.fbx", 'wb') as f:
                    f.write(fbx_response.content)
                print(f"Downloaded: {output_prefix}.fbx")
            
            # Download USDZ (if available)
            if 'usdz' in model_urls:
                usdz_url = model_urls['usdz']
                usdz_response = requests.get(usdz_url)
                with open(f"{output_prefix}.usdz", 'wb') as f:
                    f.write(usdz_response.content)
                print(f"Downloaded: {output_prefix}.usdz")
            
            # Download USDZ (if available)
            if 'obj' in model_urls:
                obj_url = model_urls['obj']
                obj_response = requests.get(obj_url)
                with open(f"{output_prefix}.obj", 'wb') as f:
                    f.write(obj_response.content)
                print(f"Downloaded: {output_prefix}.obj")

                
            return status_data
            
        elif status == "FAILED":
            print("Generation failed!")
            print(f"Error: {status_data.get('error', 'Unknown error')}")
            return None
            
        elif status in ["PENDING", "IN_PROGRESS"]:
            time.sleep(10)  # Wait 10 seconds before checking again
        else:
            print(f"Unknown status: {status}")
            time.sleep(10)

# Complete workflow
if __name__ == "__main__":
    # Path to your image
    image_path = "outputs/sam3_filled_bg_removed_org25_TH2-7_d19_LabA.jpg"
    
    # Create task
    task_id = create_image_to_3d_task(image_path, enable_pbr=True)
    
    # Wait and download
    result = wait_and_download(task_id, output_prefix="organoid_timepoints_3")
    
    if result:
        print("\n3D Model Generation Complete!")
        print(f"Task ID: {task_id}")
