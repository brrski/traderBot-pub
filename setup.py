from packaging import version
import subprocess
import sys
import pkg_resources
import json
from datetime import datetime

def get_latest_version(package_name):
    """Get the latest version of a package from PyPI"""
    try:
        output = subprocess.check_output([sys.executable, "-m", "pip", "index", "versions", package_name])
        versions = output.decode().split('\n')[0]
        if '(' in versions:
            return versions.split('(')[1].split(')')[0]
        return None
    except:
        return None

def setup_environment():
    """Install or upgrade required packages using pip"""
    
    # Define required packages with their minimum versions
    packages = {
        'pandas': '2.0.0',
        'torch': '2.0.0',
        'transformers': '4.30.0',
        'yfinance': '0.2.30',
        'requests': '2.31.0',
        'PyQt6': '6.4.0',
        'plotly': '5.13.0',
        'packaging': '23.0'
    }
    
    # Create a log dictionary
    log = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'packages': {}
    }
    
    print("Checking and installing packages...\n")
    
    for package_name, min_version in packages.items():
        print(f"\nProcessing {package_name}...")
        try:
            # Check if package is installed
            installed = pkg_resources.get_distribution(package_name)
            installed_version = installed.version
            print(f"Currently installed version: {installed_version}")
            
            # Get latest version from PyPI
            latest_version = get_latest_version(package_name)
            if latest_version:
                print(f"Latest available version: {latest_version}")
            
            # Check if upgrade is needed
            needs_upgrade = False
            if latest_version and version.parse(installed_version) < version.parse(latest_version):
                needs_upgrade = True
            elif version.parse(installed_version) < version.parse(min_version):
                needs_upgrade = True
            
            if needs_upgrade:
                print(f"Upgrading {package_name}...")
                subprocess.check_call([
                    sys.executable, 
                    "-m", 
                    "pip", 
                    "install", 
                    f"{package_name}=={latest_version or min_version}",
                    "--upgrade"
                ])
                new_version = pkg_resources.get_distribution(package_name).version
                print(f"Upgraded to version {new_version}")
            else:
                print(f"{package_name} is up to date")
            
            # Log the result
            log['packages'][package_name] = {
                'initial_version': installed_version,
                'final_version': pkg_resources.get_distribution(package_name).version,
                'latest_available': latest_version,
                'status': 'upgraded' if needs_upgrade else 'up_to_date'
            }
            
        except pkg_resources.DistributionNotFound:
            print(f"{package_name} not found. Installing...")
            try:
                # Install the latest version
                subprocess.check_call([
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    package_name
                ])
                installed_version = pkg_resources.get_distribution(package_name).version
                print(f"Installed version {installed_version}")
                
                # Log the installation
                log['packages'][package_name] = {
                    'initial_version': None,
                    'final_version': installed_version,
                    'latest_available': get_latest_version(package_name),
                    'status': 'installed'
                }
                
            except subprocess.CalledProcessError as e:
                print(f"Error installing {package_name}: {str(e)}")
                log['packages'][package_name] = {
                    'status': 'error',
                    'error_message': str(e)
                }
                return False
        except Exception as e:
            print(f"Error processing {package_name}: {str(e)}")
            log['packages'][package_name] = {
                'status': 'error',
                'error_message': str(e)
            }
    
    # Save installation log
    log_filename = f"package_installation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_filename, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nInstallation log saved to {log_filename}")
    
    # Verify installations
    print("\nVerifying installations...")
    try:
        import pandas
        import torch
        import transformers
        import yfinance
        import requests
        import PyQt6
        import plotly
        
        print("\nAll packages installed successfully!")
        print(f"Pandas version: {pandas.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Transformers version: {transformers.__version__}")
        print(f"yfinance version: {yfinance.__version__}")
        print(f"requests version: {requests.__version__}")
        print(f"PyQt6 version: {PyQt6.QtCore.QT_VERSION_STR}")
        print(f"plotly version: {plotly.__version__}")
        
        return True
    except ImportError as e:
        print(f"Error verifying installations: {str(e)}")
        return False

if __name__ == "__main__":
    setup_environment()
