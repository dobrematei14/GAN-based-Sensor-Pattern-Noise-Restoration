import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_cgan import test_cgan_training

def run_all_tests():
    """
    Run cGAN training test.
    """
    print("\n" + "="*50)
    print("TESTING cGAN TRAINING")
    print("="*50)
    
    # Test cGAN Training
    print("\nTesting cGAN Training")
    print("-"*30)
    cgan_success = test_cgan_training()
    
    # Report results
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    print(f"cGAN Training: {'✓' if cgan_success else '✗'}")
    
    print("\nOverall Status:", "✓ Test passed!" if cgan_success else "✗ Test failed")
    print("="*50)

if __name__ == "__main__":
    run_all_tests() 