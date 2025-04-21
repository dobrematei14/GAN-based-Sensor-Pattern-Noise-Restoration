import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cGAN.cGAN import train_cgan

def test_cgan_training():
    """
    Test cGAN training with minimal configuration.
    """
    # Test configuration with minimal settings
    config = {
        'batch_size': 4,        # Small batch size for testing
        'num_epochs': 2,        # Just 2 epochs for testing
        'learning_rate': 0.0002,
        'num_workers': 0        # No parallel workers for testing
    }
    
    try:
        train_cgan(config)
        print("✓ cGAN training test completed successfully")
        return True
    except Exception as e:
        print(f"✗ Error during cGAN training: {str(e)}")
        return False

if __name__ == "__main__":
    test_cgan_training() 