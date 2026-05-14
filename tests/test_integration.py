"""
Nexum v2.0 Integration Tests
Tests integration between all system modules
"""

import asyncio
import pytest
import time
import numpy as np
from unittest.mock import Mock, patch

# Import system components
from nexum.core import NexumSystem, SystemConfig
from kernel import SystemManager, CommunicationProtocol, HardwareInterface
from hardware import MotorController, BiomimeticEars, MechanicalController
from vision import ImageProcessor
from ai import LocalModelManager
from hardware.structure import BodyDesigner


class TestSystemIntegration:
    """Test integration between all system modules"""
    
    @pytest.fixture
    async def test_system(self):
        """Create test system configuration"""
        config = SystemConfig(
            log_level="DEBUG",
            enable_motors=True,
            enable_audio=True,
            enable_vision=True,
            enable_ai=True,
            enable_gpu=False  # Disable GPU for testing
        )
        
        system = NexumSystem(config)
        yield system
        
        # Cleanup
        await system.stop()
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, test_system):
        """Test complete system initialization"""
        # Initialize system
        success = await test_system.initialize()
        assert success, "System should initialize successfully"
        assert test_system.state.value == "running", "System should be in running state"
        
        # Check core components
        assert test_system.system_manager is not None, "System manager should be initialized"
        assert test_system.communication is not None, "Communication protocol should be initialized"
        assert test_system.hardware_interface is not None, "Hardware interface should be initialized"
        
        # Check hardware modules
        assert test_system.motor_controller is not None, "Motor controller should be initialized"
        assert test_system.biomimetic_ears is not None, "Biomimetic ears should be initialized"
        assert test_system.mechanical_controller is not None, "Mechanical controller should be initialized"
        
        # Check processing modules
        assert test_system.image_processor is not None, "Image processor should be initialized"
        assert test_system.ai_manager is not None, "AI manager should be initialized"
        
        # Check design module
        assert test_system.body_designer is not None, "Body designer should be initialized"
    
    @pytest.mark.asyncio
    async def test_system_status(self, test_system):
        """Test system status reporting"""
        await test_system.initialize()
        
        status = test_system.get_status()
        
        # Check status structure
        assert "state" in status, "Status should contain state"
        assert "uptime" in status, "Status should contain uptime"
        assert "total_operations" in status, "Status should contain operations count"
        assert "errors" in status, "Status should contain error count"
        assert "components" in status, "Status should contain component status"
        
        # Check component status
        components = status["components"]
        assert "system_manager" in components, "Should have system manager status"
        assert "motor_controller" in components, "Should have motor controller status"
        assert "image_processor" in components, "Should have image processor status"
        assert "ai_manager" in components, "Should have AI manager status"
    
    @pytest.mark.asyncio
    async def test_command_execution(self, test_system):
        """Test system command execution"""
        await test_system.initialize()
        
        # Test status command
        result = await test_system.execute_command("status")
        assert result["success"], "Status command should succeed"
        assert "data" in result, "Status command should return data"
        
        # Test design command
        result = await test_system.execute_command("design_body", {
            "height": 1.75,
            "weight_target": 75
        })
        assert result["success"], "Design command should succeed"
        assert "data" in result, "Design command should return data"
        
        design_data = result["data"]
        assert "total_height" in design_data, "Design should contain height"
        assert "total_weight" in design_data, "Design should contain weight"
        assert "material_usage" in design_data, "Design should contain material usage"
    
    @pytest.mark.asyncio
    async def test_motor_integration(self, test_system):
        """Test motor control integration"""
        await test_system.initialize()
        
        # Add test motor
        from hardware.kinematics.motor_controller import MotorConfig, MotorType
        motor_config = MotorConfig(
            id="test_motor",
            name="Test Motor",
            type=MotorType.SERVO,
            max_speed=10.0,
            max_acceleration=5.0,
            max_torque=2.0,
            gear_ratio=10.0,
            encoder_resolution=4096,
            min_position=-3.14159,
            max_position=3.14159,
            home_position=0.0
        )
        
        success = test_system.motor_controller.add_motor(motor_config)
        assert success, "Should be able to add motor"
        
        # Test motor movement
        result = await test_system.execute_command("move_motor", {
            "motor_id": "test_motor",
            "position": 1.5
        })
        assert result["success"], "Should be able to move motor"
    
    @pytest.mark.asyncio
    async def test_audio_integration(self, test_system):
        """Test audio processing integration"""
        await test_system.initialize()
        
        # Test audio system
        assert test_system.biomimetic_ears.is_listening(), "Audio system should be listening"
        
        # Test audio statistics
        stats = test_system.biomimetic_ears.get_statistics()
        assert "frames_processed" in stats, "Should have frame statistics"
        assert "voice_activity_ratio" in stats, "Should have voice activity statistics"
        assert "current_direction" in stats, "Should have direction estimation"
    
    @pytest.mark.asyncio
    async def test_vision_integration(self, test_system):
        """Test vision processing integration"""
        await test_system.initialize()
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test image processing
        processed = test_system.image_processor.process_image(test_image, ["enhance_contrast"])
        assert processed is not None, "Should be able to process image"
        assert processed.data.shape == test_image.shape, "Processed image should have same shape"
        
        # Test feature extraction
        features = test_system.image_processor.extract_features(test_image, ["histogram"])
        assert "histogram" in features, "Should extract histogram features"
    
    @pytest.mark.asyncio
    async def test_ai_integration(self, test_system):
        """Test AI processing integration"""
        await test_system.initialize()
        
        # Test AI manager
        available_models = test_system.ai_manager.get_available_models()
        assert len(available_models) > 0, "Should have available models"
        
        # Test model loading
        if available_models:
            model_id = available_models[0]
            success = test_system.ai_manager.load_model(model_id)
            # Note: This might fail if model files don't exist in test environment
            
            # Test model status
            status = test_system.ai_manager.get_model_status(model_id)
            assert status is not None, "Should get model status"
    
    @pytest.mark.asyncio
    async def test_mechanical_integration(self, test_system):
        """Test mechanical control integration"""
        await test_system.initialize()
        
        # Add test segment
        success = test_system.mechanical_controller.add_segment(
            "test_segment",
            mass=10.0,
            dimensions=(0.2, 0.3, 0.1),
            position=np.array([0.0, 0.5, 0.0])
        )
        assert success, "Should be able to add mechanical segment"
        
        # Test segment position
        position = test_system.mechanical_controller.get_segment_position("test_segment")
        assert position is not None, "Should get segment position"
        assert len(position) == 3, "Position should be 3D"
        
        # Test center of mass
        com = test_system.mechanical_controller.get_center_of_mass()
        assert len(com) == 3, "Center of mass should be 3D"
    
    @pytest.mark.asyncio
    async def test_communication_integration(self, test_system):
        """Test communication system integration"""
        await test_system.initialize()
        
        # Test communication protocol
        assert test_system.communication is not None, "Communication should be initialized"
        
        # Test message sending (this would require actual network setup)
        # For testing, we'll just check the protocol exists
        assert hasattr(test_system.communication, 'send_message'), "Should have send_message method"
        assert hasattr(test_system.communication, 'initialize'), "Should have initialize method"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, test_system):
        """Test system error handling"""
        await test_system.initialize()
        
        # Test invalid command
        result = await test_system.execute_command("invalid_command")
        assert not result["success"], "Invalid command should fail"
        assert "error" in result, "Should return error message"
        
        # Test command with missing parameters
        result = await test_system.execute_command("move_motor")
        assert not result["success"], "Command without parameters should fail"
        
        # Test system statistics
        status = test_system.get_status()
        assert status["errors"] >= 0, "Error count should be non-negative"
    
    @pytest.mark.asyncio
    async def test_system_shutdown(self, test_system):
        """Test graceful system shutdown"""
        await test_system.initialize()
        
        # Verify system is running
        assert test_system.running, "System should be running"
        
        # Shutdown system
        await test_system.stop()
        
        # Verify shutdown
        assert not test_system.running, "System should not be running"
        assert test_system.state.value == "stopped", "System should be in stopped state"
    
    @pytest.mark.asyncio
    async def test_resource_monitoring(self, test_system):
        """Test system resource monitoring"""
        await test_system.initialize()
        
        # Wait a bit for monitoring to collect data
        await asyncio.sleep(2)
        
        # Check system manager status
        sys_status = test_system.system_manager.get_system_status()
        assert "resources" in sys_status, "Should have resource information"
        
        resources = sys_status["resources"]
        assert "cpu_usage" in resources, "Should monitor CPU usage"
        assert "memory_usage" in resources, "Should monitor memory usage"
        assert "available_memory_gb" in resources, "Should monitor available memory"


class TestComponentCommunication:
    """Test communication between components"""
    
    @pytest.mark.asyncio
    async def test_hardware_ai_communication(self):
        """Test communication between hardware and AI components"""
        # Create minimal system for testing
        config = SystemConfig(
            enable_motors=True,
            enable_ai=True,
            enable_audio=False,
            enable_vision=False
        )
        
        system = NexumSystem(config)
        await system.initialize()
        
        try:
            # Test that hardware can send data to AI
            # This is a conceptual test - actual implementation would depend on specific interfaces
            
            # Verify both components are running
            assert system.motor_controller is not None
            assert system.ai_manager is not None
            
            # Test component status communication
            motor_status = system.motor_controller.get_all_status()
            ai_status = system.ai_manager.get_all_status()
            
            assert isinstance(motor_status, dict), "Motor status should be dictionary"
            assert isinstance(ai_status, dict), "AI status should be dictionary"
            
        finally:
            await system.stop()
    
    @pytest.mark.asyncio
    async def test_vision_ai_communication(self):
        """Test communication between vision and AI components"""
        config = SystemConfig(
            enable_vision=True,
            enable_ai=True,
            enable_motors=False,
            enable_audio=False
        )
        
        system = NexumSystem(config)
        await system.initialize()
        
        try:
            # Test that vision can send processed images to AI
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Process image
            processed = system.image_processor.process_image(test_image)
            
            # Extract features (simulating AI input)
            features = system.image_processor.extract_features(test_image)
            
            assert processed is not None, "Image should be processed"
            assert features is not None, "Features should be extracted"
            
        finally:
            await system.stop()


class TestPerformance:
    """Test system performance under load"""
    
    @pytest.mark.asyncio
    async def test_system_performance(self):
        """Test system performance with concurrent operations"""
        config = SystemConfig(
            enable_motors=True,
            enable_audio=True,
            enable_vision=True,
            enable_ai=True
        )
        
        system = NexumSystem(config)
        await system.initialize()
        
        try:
            # Measure initial status
            start_status = system.get_status()
            start_ops = start_status["total_operations"]
            
            # Run concurrent operations
            tasks = []
            
            # Simulate motor operations
            for i in range(10):
                task = system.execute_command("move_motor", {
                    "motor_id": f"motor_{i}",
                    "position": float(i)
                })
                tasks.append(task)
            
            # Simulate design operations
            for i in range(5):
                task = system.execute_command("design_body", {
                    "height": 1.7 + i * 0.1,
                    "weight_target": 70 + i * 5
                })
                tasks.append(task)
            
            # Wait for all operations to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check results
            successful_ops = sum(1 for r in results if isinstance(r, dict) and r.get("success", False))
            
            # Verify system handled load
            end_status = system.get_status()
            end_ops = end_status["total_operations"]
            
            assert end_ops > start_ops, "Operations count should increase"
            assert successful_ops > 0, "Some operations should succeed"
            
        finally:
            await system.stop()


if __name__ == "__main__":
    # Run tests manually
    pytest.main([__file__, "-v"])
