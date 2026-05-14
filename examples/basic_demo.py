"""
Nexum v2.0 Basic Demo
Demonstrates core system capabilities
"""

import asyncio
import numpy as np
import cv2
import time
from nexum.core import NexumSystem, SystemConfig


async def basic_system_demo():
    """Basic system demonstration"""
    print("=== Nexum v2.0 Basic Demo ===\n")
    
    # Create system configuration
    config = SystemConfig(
        log_level="INFO",
        enable_gpu=True,
        enable_motors=True,
        enable_audio=True,
        enable_vision=True,
        enable_ai=True
    )
    
    # Create and initialize system
    print("1. Initializing Nexum v2.0 System...")
    nexum = NexumSystem(config)
    
    if not await nexum.initialize():
        print("❌ System initialization failed")
        return
    
    print("✅ System initialized successfully")
    
    try:
        # Demo 1: System Status
        print("\n2. Checking System Status...")
        status = nexum.get_status()
        print(f"   State: {status['state']}")
        print(f"   Uptime: {status['uptime']:.2f} seconds")
        print(f"   Components: {len([c for c in status['components'].values() if c is not None])}")
        
        # Demo 2: Body Design
        print("\n3. Designing Robotic Body...")
        design_result = await nexum.execute_command("design_body", {
            "height": 1.75,
            "weight_target": 75
        })
        
        if design_result["success"]:
            design = design_result["data"]
            print(f"   ✅ Body designed: {design['total_height']:.2f}m, {design['total_weight']:.2f}kg")
            print(f"   📊 Materials: {', '.join(design['material_usage'].keys())}")
        else:
            print(f"   ❌ Body design failed: {design_result.get('error')}")
        
        # Demo 3: Motor Control (simulated)
        print("\n4. Testing Motor Control...")
        motor_result = await nexum.execute_command("move_motor", {
            "motor_id": "demo_motor",
            "position": 1.5
        })
        
        if motor_result["success"]:
            print("   ✅ Motor command executed")
        else:
            print(f"   ⚠️  Motor simulation: {motor_result.get('error', 'No motors configured')}")
        
        # Demo 4: Audio Processing
        print("\n5. Testing Audio Processing...")
        if nexum.biomimetic_ears:
            print("   🎵 Audio system active")
            stats = nexum.biomimetic_ears.get_statistics()
            print(f"   📊 Frames processed: {stats['frames_processed']}")
            print(f"   🎯 Current direction: {stats['current_direction']:.2f} rad")
        else:
            print("   ⚠️  Audio system not available")
        
        # Demo 5: Vision Processing
        print("\n6. Testing Vision Processing...")
        if nexum.image_processor:
            # Create test image
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Process image
            processed = nexum.image_processor.process_image(test_image, ["enhance_contrast"])
            
            if processed:
                print("   ✅ Image processed successfully")
                print(f"   📏 Original: {test_image.shape} → Processed: {processed.data.shape}")
                
                # Extract features
                features = nexum.image_processor.extract_features(test_image, ["histogram"])
                if "histogram" in features:
                    print(f"   📊 Features extracted: histogram ({len(features['histogram'])} dimensions)")
            else:
                print("   ❌ Image processing failed")
        else:
            print("   ⚠️  Vision system not available")
        
        # Demo 6: AI Processing
        print("\n7. Testing AI Processing...")
        if nexum.ai_manager:
            available_models = nexum.ai_manager.get_available_models()
            print(f"   🤖 Available models: {len(available_models)}")
            
            if available_models:
                model_id = available_models[0]
                print(f"   🧠 Testing model: {model_id}")
                
                # Try to load model (may fail if model files don't exist)
                load_success = nexum.ai_manager.load_model(model_id)
                if load_success:
                    print("   ✅ Model loaded successfully")
                    
                    # Try inference (with dummy data)
                    inference_result = await nexum.execute_command("ai_inference", {
                        "model_id": model_id,
                        "inputs": "Hello, Nexum!"
                    })
                    
                    if inference_result["success"]:
                        print("   ✅ AI inference completed")
                    else:
                        print(f"   ⚠️  AI inference: {inference_result.get('error')}")
                else:
                    print("   ⚠️  Model loading failed (model files may not exist)")
            else:
                print("   ⚠️  No AI models available")
        else:
            print("   ⚠️  AI system not available")
        
        # Demo 7: System Monitoring
        print("\n8. System Monitoring...")
        print("   📊 Monitoring system resources...")
        
        # Let system run for a few seconds to collect data
        await asyncio.sleep(3)
        
        final_status = nexum.get_status()
        resources = final_status["components"]["system_manager"]["resources"] if final_status["components"]["system_manager"] else {}
        
        if resources:
            print(f"   💾 Memory: {resources.get('memory_usage', 0):.1f}%")
            print(f"   🖥️  CPU: {resources.get('cpu_usage', 0):.1f}%")
            print(f"   🎮 GPU: {resources.get('gpu_usage', 0):.1f}%")
        
        print(f"   📈 Operations: {final_status['total_operations']}")
        print(f"   ⚠️  Errors: {final_status['errors']}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
    
    finally:
        # Cleanup
        print("\n9. Shutting down system...")
        await nexum.stop()
        print("✅ System shutdown complete")


async def interactive_demo():
    """Interactive demo with user input"""
    print("=== Nexum v2.0 Interactive Demo ===\n")
    
    config = SystemConfig(log_level="INFO")
    nexum = NexumSystem(config)
    
    if not await nexum.initialize():
        print("❌ System initialization failed")
        return
    
    print("✅ System ready for interactive commands")
    print("Available commands:")
    print("  status    - Show system status")
    print("  design    - Design robotic body")
    print("  motors    - Test motor control")
    print("  audio     - Test audio system")
    print("  vision    - Test vision system")
    print("  ai        - Test AI system")
    print("  quit      - Exit demo")
    
    try:
        while True:
            command = input("\nnexum> ").strip().lower()
            
            if command == "quit" or command == "exit":
                break
            elif command == "status":
                status = nexum.get_status()
                print(f"System State: {status['state']}")
                print(f"Uptime: {status['uptime']:.2f}s")
                print(f"Operations: {status['total_operations']}")
            
            elif command == "design":
                result = await nexum.execute_command("design_body", {
                    "height": 1.75,
                    "weight_target": 75
                })
                if result["success"]:
                    design = result["data"]
                    print(f"Body: {design['total_height']:.2f}m, {design['total_weight']:.2f}kg")
                else:
                    print(f"Design failed: {result.get('error')}")
            
            elif command == "motors":
                result = await nexum.execute_command("move_motor", {
                    "motor_id": "test",
                    "position": 1.0
                })
                if result["success"]:
                    print("Motor command executed")
                else:
                    print(f"Motor error: {result.get('error')}")
            
            elif command == "audio":
                if nexum.biomimetic_ears:
                    stats = nexum.biomimetic_ears.get_statistics()
                    print(f"Audio frames: {stats['frames_processed']}")
                    print(f"Voice activity: {stats['voice_activity_ratio']:.2%}")
                else:
                    print("Audio system not available")
            
            elif command == "vision":
                if nexum.image_processor:
                    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    processed = nexum.image_processor.process_image(test_image)
                    if processed:
                        print("Image processed successfully")
                        print(f"Shape: {processed.data.shape}")
                    else:
                        print("Image processing failed")
                else:
                    print("Vision system not available")
            
            elif command == "ai":
                if nexum.ai_manager:
                    models = nexum.ai_manager.get_available_models()
                    print(f"Available models: {len(models)}")
                    if models:
                        print(f"Models: {', '.join(models)}")
                    else:
                        print("No models available")
                else:
                    print("AI system not available")
            
            else:
                print(f"Unknown command: {command}")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    
    finally:
        await nexum.stop()


async def performance_demo():
    """Performance testing demo"""
    print("=== Nexum v2.0 Performance Demo ===\n")
    
    config = SystemConfig(log_level="WARNING")  # Reduce log noise
    nexum = NexumSystem(config)
    
    if not await nexum.initialize():
        print("❌ System initialization failed")
        return
    
    try:
        print("🚀 Running performance tests...\n")
        
        # Test 1: System Response Time
        print("1. Testing System Response Time...")
        start_time = time.time()
        
        for i in range(100):
            await nexum.execute_command("status")
        
        response_time = (time.time() - start_time) / 100
        print(f"   Average response time: {response_time*1000:.2f}ms")
        
        # Test 2: Design Performance
        print("\n2. Testing Design Performance...")
        start_time = time.time()
        
        for i in range(10):
            await nexum.execute_command("design_body", {
                "height": 1.7 + i * 0.05,
                "weight_target": 70 + i * 2
            })
        
        design_time = (time.time() - start_time) / 10
        print(f"   Average design time: {design_time:.3f}s")
        
        # Test 3: Image Processing Performance
        if nexum.image_processor:
            print("\n3. Testing Image Processing Performance...")
            test_images = [
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(50)
            ]
            
            start_time = time.time()
            for img in test_images:
                nexum.image_processor.process_image(img, ["enhance_contrast"])
            
            processing_time = (time.time() - start_time) / 50
            fps = 1.0 / processing_time
            print(f"   Average processing time: {processing_time:.3f}s")
            print(f"   Theoretical FPS: {fps:.1f}")
        
        # Test 4: Memory Usage
        print("\n4. Testing Memory Usage...")
        initial_status = nexum.get_status()
        
        # Run intensive operations
        for i in range(20):
            await nexum.execute_command("design_body", {
                "height": 1.8,
                "weight_target": 80
            })
        
        final_status = nexum.get_status()
        
        if final_status["components"]["system_manager"]:
            resources = final_status["components"]["system_manager"]["resources"]
            memory_usage = resources.get('memory_usage', 0)
            print(f"   Memory usage after load: {memory_usage:.1f}%")
        
        # Test 5: Concurrent Operations
        print("\n5. Testing Concurrent Operations...")
        start_time = time.time()
        
        tasks = []
        for i in range(50):
            task = nexum.execute_command("status")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        concurrent_time = time.time() - start_time
        successful_ops = sum(1 for r in results if isinstance(r, dict) and r.get("success", False))
        
        print(f"   {successful_ops}/50 operations successful")
        print(f"   Concurrent execution time: {concurrent_time:.3f}s")
        print(f"   Operations per second: {successful_ops/concurrent_time:.1f}")
        
        print("\n✅ Performance tests completed!")
        
    except Exception as e:
        print(f"❌ Performance test error: {e}")
    
    finally:
        await nexum.stop()


async def main():
    """Main demo entry point"""
    print("Nexum v2.0 Demo Suite")
    print("Select demo mode:")
    print("1. Basic Demo")
    print("2. Interactive Demo")
    print("3. Performance Demo")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            await basic_system_demo()
        elif choice == "2":
            await interactive_demo()
        elif choice == "3":
            await performance_demo()
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\nDemo cancelled by user")


if __name__ == "__main__":
    asyncio.run(main())
