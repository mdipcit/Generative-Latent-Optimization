#!/usr/bin/env python3
"""
Unit Tests for OptimizationConfig Module

Comprehensive testing of optimization configuration functionality
including parameter validation, serialization, and edge cases.
"""

import pytest
import json
import tempfile
from pathlib import Path
from dataclasses import asdict

# Add project paths
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.generative_latent_optimization.optimization.latent_optimizer import OptimizationConfig
from tests.fixtures.test_helpers import print_test_header, print_test_result


class TestOptimizationConfig:
    """Test suite for OptimizationConfig class"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        print_test_header("Default Configuration Test")
        
        config = OptimizationConfig()
        
        # Verify all default values
        assert config.iterations == 150
        assert config.learning_rate == 0.4
        assert config.loss_function == 'mse'
        assert config.convergence_threshold == 1e-6
        assert config.checkpoint_interval == 20
        assert config.device == "cuda"
        
        print_test_result("Default values", True, "All defaults correct")
    
    def test_custom_configuration(self):
        """Test custom configuration values"""
        print_test_header("Custom Configuration Test")
        
        config = OptimizationConfig(
            iterations=100,
            learning_rate=0.1,
            loss_function='l1',
            convergence_threshold=1e-5,
            checkpoint_interval=10,
            device='cpu'
        )
        
        assert config.iterations == 100
        assert config.learning_rate == 0.1
        assert config.loss_function == 'l1'
        assert config.convergence_threshold == 1e-5
        assert config.checkpoint_interval == 10
        assert config.device == 'cpu'
        
        print_test_result("Custom values", True, "All custom values set correctly")
    
    def test_parameter_ranges_validation(self):
        """Test parameter range validation (logical bounds)"""
        print_test_header("Parameter Range Validation Test")
        
        # Test valid ranges
        valid_configs = [
            OptimizationConfig(iterations=1),           # Minimum iterations
            OptimizationConfig(iterations=1000),        # High iterations
            OptimizationConfig(learning_rate=1e-6),     # Very small LR
            OptimizationConfig(learning_rate=10.0),     # Large LR
            OptimizationConfig(convergence_threshold=1e-10),  # Very strict
            OptimizationConfig(convergence_threshold=1e-2),   # Loose threshold
            OptimizationConfig(checkpoint_interval=1),  # Every iteration
            OptimizationConfig(checkpoint_interval=100) # Sparse checkpoints
        ]
        
        for i, config in enumerate(valid_configs):
            assert isinstance(config, OptimizationConfig)
            print_test_result(f"Valid config {i+1}", True, "Created successfully")
        
        # Test boundary cases that should work
        boundary_configs = [
            OptimizationConfig(iterations=0),           # Zero iterations (edge case)
            OptimizationConfig(learning_rate=0.0),      # Zero learning rate
        ]
        
        for i, config in enumerate(boundary_configs):
            assert isinstance(config, OptimizationConfig)
            print_test_result(f"Boundary config {i+1}", True, "Handled gracefully")
    
    def test_loss_function_validation(self):
        """Test loss function parameter validation"""
        print_test_header("Loss Function Validation Test")
        
        # Test valid loss functions
        valid_loss_functions = ['mse', 'l1']
        
        for loss_func in valid_loss_functions:
            config = OptimizationConfig(loss_function=loss_func)
            assert config.loss_function == loss_func
            print_test_result(f"Loss function '{loss_func}'", True, "Valid function accepted")
        
        # Test that invalid loss functions are stored but would be caught at runtime
        # (The config itself doesn't validate, the optimizer does)
        invalid_config = OptimizationConfig(loss_function='invalid_loss')
        assert invalid_config.loss_function == 'invalid_loss'
        print_test_result("Invalid loss function storage", True, 
                         "Config stores value (validation happens at runtime)")
    
    def test_device_parameter(self):
        """Test device parameter handling"""
        print_test_header("Device Parameter Test")
        
        # Test valid device strings
        valid_devices = ['cpu', 'cuda', 'cuda:0', 'cuda:1']
        
        for device in valid_devices:
            config = OptimizationConfig(device=device)
            assert config.device == device
            print_test_result(f"Device '{device}'", True, "Device string accepted")
        
        # Test that arbitrary device strings are stored
        arbitrary_device = OptimizationConfig(device='some_device')
        assert arbitrary_device.device == 'some_device'
        print_test_result("Arbitrary device", True, "Arbitrary string stored")
    
    def test_configuration_serialization(self):
        """Test configuration serialization and deserialization"""
        print_test_header("Configuration Serialization Test")
        
        # Create test configuration
        original_config = OptimizationConfig(
            iterations=75,
            learning_rate=0.25,
            loss_function='l1',
            convergence_threshold=5e-6,
            checkpoint_interval=15,
            device='cpu'
        )
        
        # Convert to dictionary
        config_dict = asdict(original_config)
        
        # Verify dictionary contains all fields
        expected_fields = [
            'iterations', 'learning_rate', 'loss_function', 
            'convergence_threshold', 'checkpoint_interval', 'device'
        ]
        
        for field in expected_fields:
            assert field in config_dict
        
        print_test_result("Dict conversion", True, "All fields present")
        
        # Test JSON serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            temp_path = f.name
        
        try:
            # Read back from JSON
            with open(temp_path, 'r') as f:
                loaded_dict = json.load(f)
            
            # Create new config from loaded dictionary
            restored_config = OptimizationConfig(**loaded_dict)
            
            # Verify all fields match
            assert restored_config.iterations == original_config.iterations
            assert restored_config.learning_rate == original_config.learning_rate
            assert restored_config.loss_function == original_config.loss_function
            assert restored_config.convergence_threshold == original_config.convergence_threshold
            assert restored_config.checkpoint_interval == original_config.checkpoint_interval
            assert restored_config.device == original_config.device
            
            print_test_result("JSON serialization", True, "Round-trip successful")
            
        finally:
            # Cleanup
            Path(temp_path).unlink()
    
    def test_configuration_equality(self):
        """Test configuration equality comparison"""
        print_test_header("Configuration Equality Test")
        
        config1 = OptimizationConfig(iterations=100, learning_rate=0.1)
        config2 = OptimizationConfig(iterations=100, learning_rate=0.1)
        config3 = OptimizationConfig(iterations=100, learning_rate=0.2)
        
        # Test equality
        assert config1 == config2
        print_test_result("Equal configs", True, "Correctly identified as equal")
        
        # Test inequality
        assert config1 != config3
        print_test_result("Different configs", True, "Correctly identified as different")
    
    def test_configuration_copy_and_modification(self):
        """Test configuration copying and modification"""
        print_test_header("Configuration Copy/Modify Test")
        
        original = OptimizationConfig(iterations=100, learning_rate=0.1)
        
        # Test shallow copy
        from copy import copy
        copied = copy(original)
        
        assert copied == original
        assert copied is not original
        print_test_result("Shallow copy", True, "Copy created correctly")
        
        # Modify copy
        copied.iterations = 200
        
        assert copied.iterations == 200
        assert original.iterations == 100  # Original unchanged
        print_test_result("Independent modification", True, "Original config preserved")
    
    def test_configuration_field_types(self):
        """Test field type consistency"""
        print_test_header("Field Type Consistency Test")
        
        config = OptimizationConfig()
        
        # Verify field types
        assert isinstance(config.iterations, int)
        assert isinstance(config.learning_rate, float)
        assert isinstance(config.loss_function, str)
        assert isinstance(config.convergence_threshold, float)
        assert isinstance(config.checkpoint_interval, int)
        assert isinstance(config.device, str)
        
        print_test_result("Field types", True, "All types correct")
    
    def test_extreme_parameter_values(self):
        """Test extreme but valid parameter values"""
        print_test_header("Extreme Parameter Values Test")
        
        # Test very large iterations
        config_large = OptimizationConfig(iterations=10000)
        assert config_large.iterations == 10000
        print_test_result("Large iterations", True, "10000 iterations accepted")
        
        # Test very small learning rate
        config_small_lr = OptimizationConfig(learning_rate=1e-10)
        assert config_small_lr.learning_rate == 1e-10
        print_test_result("Tiny learning rate", True, "1e-10 learning rate accepted")
        
        # Test very small convergence threshold
        config_strict = OptimizationConfig(convergence_threshold=1e-15)
        assert config_strict.convergence_threshold == 1e-15
        print_test_result("Strict convergence", True, "1e-15 threshold accepted")
        
        # Test large checkpoint interval
        config_sparse = OptimizationConfig(checkpoint_interval=1000)
        assert config_sparse.checkpoint_interval == 1000
        print_test_result("Sparse checkpoints", True, "1000 interval accepted")
    
    def test_configuration_string_representation(self):
        """Test string representation of configuration"""
        print_test_header("String Representation Test")
        
        config = OptimizationConfig(
            iterations=50,
            learning_rate=0.2,
            loss_function='l1'
        )
        
        config_str = str(config)
        
        # Verify string contains key information
        assert 'iterations=50' in config_str
        assert 'learning_rate=0.2' in config_str
        assert "loss_function='l1'" in config_str
        
        print_test_result("String representation", True, "Contains all key fields")
    
    def test_configuration_dictionary_creation(self):
        """Test creating configurations from dictionaries"""
        print_test_header("Dictionary Creation Test")
        
        config_dict = {
            'iterations': 80,
            'learning_rate': 0.3,
            'loss_function': 'mse',
            'convergence_threshold': 2e-6,
            'checkpoint_interval': 25,
            'device': 'cuda:1'
        }
        
        config = OptimizationConfig(**config_dict)
        
        # Verify all values were set correctly
        assert config.iterations == 80
        assert config.learning_rate == 0.3
        assert config.loss_function == 'mse'
        assert config.convergence_threshold == 2e-6
        assert config.checkpoint_interval == 25
        assert config.device == 'cuda:1'
        
        print_test_result("Dictionary creation", True, "All values set from dict")
    
    def test_partial_configuration_override(self):
        """Test partial configuration override"""
        print_test_header("Partial Override Test")
        
        # Create config with only some parameters specified
        partial_config = OptimizationConfig(
            iterations=200,
            learning_rate=0.05
            # Other parameters should use defaults
        )
        
        assert partial_config.iterations == 200
        assert partial_config.learning_rate == 0.05
        assert partial_config.loss_function == 'mse'  # Default
        assert partial_config.convergence_threshold == 1e-6  # Default
        assert partial_config.checkpoint_interval == 20  # Default
        assert partial_config.device == "cuda"  # Default
        
        print_test_result("Partial override", True, "Specified values + defaults")


# Convenience function for direct execution
def run_all_tests():
    """Run all tests manually without pytest runner"""
    print("🧪 Starting OptimizationConfig Unit Tests")
    print("=" * 60)
    
    test_instance = TestOptimizationConfig()
    
    # Run all tests
    try:
        test_instance.test_default_configuration()
        test_instance.test_custom_configuration()
        test_instance.test_parameter_ranges_validation()
        test_instance.test_loss_function_validation()
        test_instance.test_device_parameter()
        test_instance.test_configuration_serialization()
        test_instance.test_configuration_equality()
        test_instance.test_configuration_copy_and_modification()
        test_instance.test_configuration_field_types()
        test_instance.test_extreme_parameter_values()
        test_instance.test_configuration_string_representation()
        test_instance.test_configuration_dictionary_creation()
        test_instance.test_partial_configuration_override()
        
        print("\n" + "=" * 60)
        print("🎉 ALL OPTIMIZATION CONFIG TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)