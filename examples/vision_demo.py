"""
Nexum v2.0 Vision Demo
Demonstrates computer vision capabilities
"""

import asyncio
import cv2
import numpy as np
import time
from nexum.core import NexumSystem, SystemConfig
from vision.processing import ImageConfig


async def vision_processing_demo():
    """Demonstrate vision processing capabilities"""
    print("=== Nexum v2.0 Vision Processing Demo ===\n")
    
    # Create system with vision enabled
    config = SystemConfig(
        log_level="INFO",
        enable_vision=True,
        enable_gpu=True,
        enable_motors=False,
        enable_audio=False,
        enable_ai=False
    )
    
    nexum = NexumSystem(config)
    
    if not await nexum.initialize():
        print("❌ System initialization failed")
        return
    
    try:
        print("✅ Vision system initialized")
        
        if not nexum.image_processor:
            print("❌ Image processor not available")
            return
        
        # Demo 1: Basic Image Processing
        print("\n1. Basic Image Processing Demo")
        await test_basic_processing(nexum.image_processor)
        
        # Demo 2: Feature Extraction
        print("\n2. Feature Extraction Demo")
        await test_feature_extraction(nexum.image_processor)
        
        # Demo 3: Real-time Processing
        print("\n3. Real-time Processing Demo")
        await test_realtime_processing(nexum.image_processor)
        
        # Demo 4: Performance Testing
        print("\n4. Performance Testing Demo")
        await test_vision_performance(nexum.image_processor)
        
        print("\n✅ Vision demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Vision demo error: {e}")
    
    finally:
        await nexum.stop()


async def test_basic_processing(image_processor):
    """Test basic image processing operations"""
    print("   Testing basic image operations...")
    
    # Create test images
    test_images = {
        "random_noise": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "gradient": create_gradient_image(480, 640),
        "circles": create_circles_image(480, 640),
        "text": create_text_image(480, 640, "Nexum v2.0")
    }
    
    operations = [
        "enhance_contrast",
        "reduce_noise:bilateral",
        "sharpen:unsharp_mask",
        "resize:800,600"
    ]
    
    for image_name, image in test_images.items():
        print(f"   📸 Processing {image_name}...")
        
        for operation in operations:
            start_time = time.time()
            
            processed = image_processor.process_image(image, [operation])
            
            processing_time = time.time() - start_time
            
            if processed:
                print(f"      ✅ {operation}: {processing_time*1000:.1f}ms")
            else:
                print(f"      ❌ {operation}: failed")
        
        print(f"      📊 Original: {image.shape} → Enhanced: {processed.data.shape if processed else 'N/A'}")


async def test_feature_extraction(image_processor):
    """Test feature extraction capabilities"""
    print("   Testing feature extraction...")
    
    # Create test image with distinct features
    test_image = create_feature_test_image(480, 640)
    
    # Test different feature types
    feature_types = [
        ["histogram"],
        ["hog"],
        ["keypoints"],
        ["histogram", "hog"],
        ["histogram", "hog", "keypoints"]
    ]
    
    for features in feature_types:
        print(f"   🔍 Extracting {', '.join(features)}...")
        
        start_time = time.time()
        extracted_features = image_processor.extract_features(test_image, features)
        extraction_time = time.time() - start_time
        
        print(f"      ⏱️  Time: {extraction_time*1000:.1f}ms")
        
        for feature_type, feature_data in extracted_features.items():
            if feature_type == "histogram":
                print(f"      📊 Histogram: {len(feature_data)} dimensions")
            elif feature_type == "hog":
                print(f"      🐗 HOG: {len(feature_data)} features")
            elif feature_type == "keypoints":
                print(f"      📍 Keypoints: {len(feature_data) if feature_data else 0}")
            elif feature_type == "descriptors":
                print(f"      📝 Descriptors: {feature_data.shape if hasattr(feature_data, 'shape') else 'N/A'}")


async def test_realtime_processing(image_processor):
    """Test real-time image processing"""
    print("   Testing real-time processing...")
    
    # Simulate camera feed
    frame_count = 0
    start_time = time.time()
    processing_times = []
    
    print("   📹 Simulating camera feed (press Ctrl+C to stop)...")
    
    try:
        for i in range(100):  # Process 100 frames
            # Generate test frame
            frame = create_test_frame(480, 640, i)
            
            # Process frame
            frame_start = time.time()
            processed = image_processor.process_image(frame, [
                "enhance_contrast",
                "reduce_noise:bilateral"
            ])
            frame_time = time.time() - frame_start
            
            processing_times.append(frame_time)
            frame_count += 1
            
            # Display progress
            if frame_count % 10 == 0:
                avg_time = sum(processing_times[-10:]) / 10
                fps = 1.0 / avg_time
                print(f"      📹 Frame {frame_count}: {frame_time*1000:.1f}ms ({fps:.1f} FPS)")
            
            # Simulate real-time timing (30 FPS)
            await asyncio.sleep(1/30)
    
    except KeyboardInterrupt:
        print("   ⏹️  Real-time test stopped by user")
    
    # Calculate statistics
    total_time = time.time() - start_time
    avg_processing_time = sum(processing_times) / len(processing_times)
    avg_fps = 1.0 / avg_processing_time
    
    print(f"   📊 Statistics:")
    print(f"      Frames processed: {frame_count}")
    print(f"      Total time: {total_time:.2f}s")
    print(f"      Average processing time: {avg_processing_time*1000:.1f}ms")
    print(f"      Average FPS: {avg_fps:.1f}")


async def test_vision_performance(image_processor):
    """Test vision system performance"""
    print("   Testing vision performance...")
    
    # Test different image sizes
    test_sizes = [
        (240, 320),
        (480, 640),
        (720, 1280),
        (1080, 1920)
    ]
    
    operations = ["enhance_contrast", "reduce_noise:bilateral"]
    
    print("   🚀 Performance benchmark:")
    print("   Resolution | Processing Time | FPS")
    print("   ----------|----------------|-----")
    
    for height, width in test_sizes:
        # Create test image
        test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Process multiple times for average
        times = []
        for _ in range(10):
            start_time = time.time()
            processed = image_processor.process_image(test_image, operations)
            processing_time = time.time() - start_time
            times.append(processing_time)
        
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        
        print(f"   {width}x{height} | {avg_time*1000:8.1f}ms | {fps:4.1f}")
    
    # Test memory usage
    print("\n   💾 Memory usage test:")
    initial_stats = image_processor.get_statistics()
    
    # Process many images
    for i in range(50):
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        image_processor.process_image(test_image, operations)
    
    final_stats = image_processor.get_statistics()
    
    print(f"      Frames processed: {final_stats['frames_processed']}")
    print(f"      Input buffer size: {final_stats['input_buffer_size']}")
    print(f"      Output buffer size: {final_stats['output_buffer_size']}")
    print(f"      GPU available: {final_stats['gpu_available']}")


def create_gradient_image(height, width):
    """Create a gradient test image"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            image[i, j] = [
                int(255 * i / height),
                int(255 * j / width),
                int(255 * (i + j) / (height + width))
            ]
    
    return image


def create_circles_image(height, width):
    """Create an image with circles"""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw circles
    cv2.circle(image, (width//4, height//4), 50, (255, 0, 0), -1)
    cv2.circle(image, (3*width//4, height//4), 50, (0, 255, 0), -1)
    cv2.circle(image, (width//2, 3*height//4), 50, (0, 0, 255), -1)
    
    return image


def create_text_image(height, width, text):
    """Create an image with text"""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 3
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Center text
    x = (width - text_width) // 2
    y = (height + text_height) // 2
    
    cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 0), thickness)
    
    return image


def create_feature_test_image(height, width):
    """Create an image with various features for testing"""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add geometric shapes
    cv2.rectangle(image, (50, 50), (200, 200), (255, 0, 0), -1)
    cv2.circle(image, (400, 150), 80, (0, 255, 0), -1)
    cv2.ellipse(image, (300, 350), (100, 50), 45, 0, 360, (0, 0, 255), -1)
    
    # Add lines
    cv2.line(image, (0, 0), (width, height), (128, 128, 128), 2)
    cv2.line(image, (width, 0), (0, height), (128, 128, 128), 2)
    
    # Add noise
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image


def create_test_frame(height, width, frame_number):
    """Create a test frame for real-time simulation"""
    # Create base image
    image = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    # Add moving element
    x = int(width * (0.5 + 0.3 * np.sin(frame_number * 0.1)))
    y = int(height * (0.5 + 0.3 * np.cos(frame_number * 0.1)))
    
    cv2.circle(image, (x, y), 30, (255, 255, 255), -1)
    
    # Add frame number
    cv2.putText(image, f"Frame {frame_number}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add noise
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image


async def main():
    """Main demo entry point"""
    print("Nexum v2.0 Vision Processing Demo")
    print("This demo showcases the computer vision capabilities of Nexum v2.0")
    print()
    
    try:
        await vision_processing_demo()
    except KeyboardInterrupt:
        print("\nDemo cancelled by user")
    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
