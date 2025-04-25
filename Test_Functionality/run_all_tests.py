import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Test_Functionality.test_cgan import test_cgan_training
from Test_Functionality.test_spn_extraction import test_spn_extraction

def run_all_tests():
    """Run all tests in the Test_Functionality directory."""
    print("\n" + "="*50)
    print("RUNNING ALL TESTS")
    print("="*50)
    
    # Test cGAN Training
    print("\nTesting cGAN Training")
    print("-"*30)
    cgan_success = test_cgan_training()
    
    # Test SPN Extraction
    print("\nTesting SPN Extraction")
    print("-"*30)
    spn_success = test_spn_extraction()
    
    # Report results
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    print(f"cGAN Training: {'✓' if cgan_success else '✗'}")
    print(f"SPN Extraction: {'✓' if spn_success else '✗'}")
    
    overall_success = cgan_success and spn_success
    print("\nOverall Status:", "✓ All tests passed!" if overall_success else "✗ Some tests failed")
    print("="*50)

if __name__ == "__main__":
    run_all_tests() 