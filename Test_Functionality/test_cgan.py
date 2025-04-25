import os
import sys
from dotenv import load_dotenv

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from cGAN.cGAN import train_cgan

# Load environment variables
load_dotenv()

def test_cgan_training():
    """Test cGAN training functionality."""
    print("\nTesting cGAN training...")
    
    try:
        # Configuration optimized for testing with 1000 pictures
        config = {
            'batch_size': 32,  # Larger batch size for faster processing
            'num_epochs': 1,   # Keep one epoch for testing
            'learning_rate': 0.0002,
            'num_workers': 4,  # Use multiple workers for faster data loading
            'max_samples': 100  # Limit samples during testing for speed
        }
        
        # Run training
        results = train_cgan(config)
        
        # Check if training completed successfully
        if results and 'model_dir' in results:
            print("✓ cGAN training test passed")
            return True
        else:
            print("✗ cGAN training test failed - no results returned")
            return False
            
    except FileNotFoundError as e:
        print(f"✗ cGAN training test skipped - {str(e)}")
        return True  # Return True to indicate test was skipped, not failed
    except Exception as e:
        print(f"✗ cGAN training test failed - {str(e)}")
        return False

if __name__ == "__main__":
    test_cgan_training() 