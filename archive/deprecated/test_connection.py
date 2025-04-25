#!/usr/bin/env python3
import sys
import os
import subprocess
import time
import socket
import psutil
import platform
import json
from urllib.request import urlopen
from urllib.error import URLError
import threading

def print_section(title):
    print("\n" + "=" * 50)
    print(f"{title}")
    print("=" * 50)

def check_port(host, port, timeout=2):
    """Test if a port is open on a given host."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        if result == 0:
            print(f"✅ Connection to {host}:{port} successful")
            # Try to identify which process is using the port
            if platform.system() in ["Darwin", "Linux"]:
                try:
                    lsof_result = subprocess.run(["lsof", "-i", f":{port}"], capture_output=True, text=True)
                    if lsof_result.stdout.strip():
                        print(f"Process using port {port}:\n{lsof_result.stdout.strip()}")
                except Exception as e:
                    print(f"Failed to get process info: {e}")
            elif platform.system() == "Windows":
                try:
                    netstat_result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
                    for line in netstat_result.stdout.splitlines():
                        if f":{port}" in line:
                            print(f"Process using port {port}: {line}")
                except Exception as e:
                    print(f"Failed to get process info: {e}")
        else:
            print(f"❌ Connection to {host}:{port} failed (Error: {result})")
            # Check if any process is listening on this port
            print(f"Checking if any process is listening on port {port}...")
            if platform.system() in ["Darwin", "Linux"]:
                try:
                    netstat_result = subprocess.run(["netstat", "-tuln"], capture_output=True, text=True)
                    for line in netstat_result.stdout.splitlines():
                        if f":{port}" in line:
                            print(f"Found listening socket: {line}")
                except Exception as e:
                    print(f"Failed to check netstat: {e}")
            elif platform.system() == "Windows":
                try:
                    netstat_result = subprocess.run(["netstat", "-an"], capture_output=True, text=True)
                    for line in netstat_result.stdout.splitlines():
                        if f":{port}" in line:
                            print(f"Found listening socket: {line}")
                except Exception as e:
                    print(f"Failed to check netstat: {e}")
    except Exception as e:
        print(f"❌ Error checking {host}:{port}: {e}")
    finally:
        try:
            sock.close()
        except:
            pass

def test_binding(port):
    """Test if we can bind to a specific port on all interfaces."""
    try:
        print(f"Testing if we can bind to port {port} on all interfaces...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', port))
        print(f"✅ Successfully bound to all interfaces on port {port}")
        sock.close()
        return True
    except Exception as e:
        print(f"❌ Failed to bind to all interfaces on port {port}: {e}")
        return False

def get_network_interfaces():
    """Get detailed information about network interfaces."""
    print_section("NETWORK INTERFACES")
    
    # Socket API interfaces
    try:
        print("Socket API available interfaces:")
        for interface in socket.if_nameindex():
            print(f"Interface: {interface}")
    except Exception as e:
        print(f"Failed to get socket interfaces: {e}")
    
    # Use ifconfig/ipconfig based on platform
    if platform.system() in ["Darwin", "Linux"]:
        try:
            print("\nNetwork interfaces (ifconfig):")
            result = subprocess.run(["ifconfig"], capture_output=True, text=True)
            print(result.stdout)
        except Exception as e:
            print(f"Failed to run ifconfig: {e}")
    elif platform.system() == "Windows":
        try:
            print("\nNetwork interfaces (ipconfig):")
            result = subprocess.run(["ipconfig"], capture_output=True, text=True)
            print(result.stdout)
        except Exception as e:
            print(f"Failed to run ipconfig: {e}")

def check_running_processes():
    """Check if specific processes are running."""
    print_section("PROCESS CHECK")
    
    # Check for enhanced_server_v5.py
    server_found = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'enhanced_server_v5.py' in ' '.join(proc.info['cmdline'] or []):
                server_found = True
                print(f"✅ Process 'enhanced_server_v5.py' is running (PID: {proc.info['pid']})")
                # Get more details about the process
                process = psutil.Process(proc.info['pid'])
                print(f"   Creation time: {time.ctime(process.create_time())}")
                print(f"   Status: {process.status()}")
                print(f"   CPU usage: {process.cpu_percent()}%")
                print(f"   Memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
                print(f"   Command line: {' '.join(process.cmdline())}")
                print(f"   Open files: {len(process.open_files())}")
                print(f"   Connections: {len(process.connections())}")
                # List connections
                for conn in process.connections():
                    print(f"     {conn.laddr.ip}:{conn.laddr.port} -> {conn.raddr.ip if conn.raddr else 'None'}:{conn.raddr.port if conn.raddr else 'None'} ({conn.status})")
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if not server_found:
        print("❌ Process 'enhanced_server_v5.py' is NOT running")
        
        # Try to find any Python processes
        print("\nListing all Python processes:")
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    print(f"Python process (PID: {proc.info['pid']}): {cmdline}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

def check_internet_connectivity():
    """Check internet connectivity by connecting to well-known sites."""
    print_section("INTERNET CONNECTIVITY")
    
    sites_to_check = [
        ("Google", "https://www.google.com"),
        ("Cloudflare", "https://1.1.1.1"),
        ("GitHub", "https://github.com"),
        ("Amazon AWS", "https://aws.amazon.com")
    ]
    
    for name, url in sites_to_check:
        try:
            with urlopen(url, timeout=5) as response:
                print(f"✅ Connected to {name} ({url}) - Status: {response.status}")
        except URLError as e:
            print(f"❌ Failed to connect to {name} ({url}): {e}")
        except Exception as e:
            print(f"❌ Error checking {name} ({url}): {e}")

def check_dns_resolution():
    """Check DNS resolution for common domains."""
    print_section("DNS RESOLUTION")
    
    domains = ["google.com", "github.com", "amazon.com", "cloudflare.com", "localhost"]
    
    for domain in domains:
        try:
            ip_address = socket.gethostbyname(domain)
            print(f"✅ Successfully resolved {domain} to {ip_address}")
        except socket.gaierror as e:
            print(f"❌ Failed to resolve {domain}: {e}")
        except Exception as e:
            print(f"❌ Error resolving {domain}: {e}")

def check_open_ports():
    """Check all open ports on the system."""
    print_section("OPEN PORTS")
    
    if platform.system() in ["Darwin", "Linux"]:
        try:
            netstat_result = subprocess.run(["netstat", "-tuln"], capture_output=True, text=True)
            print("Open ports (netstat):")
            print(netstat_result.stdout)
        except Exception as e:
            print(f"Failed to run netstat: {e}")
    elif platform.system() == "Windows":
        try:
            netstat_result = subprocess.run(["netstat", "-an"], capture_output=True, text=True)
            print("Open ports (netstat):")
            print(netstat_result.stdout)
        except Exception as e:
            print(f"Failed to run netstat: {e}")

def test_socket_connections():
    """Test socket connections to various common services."""
    print_section("SOCKET CONNECTION TESTS")
    
    # Test HTTP/HTTPS connectivity
    check_port("localhost", 80)  # HTTP
    check_port("localhost", 443)  # HTTPS
    check_port("localhost", 8080)  # Common web server port
    
    # Test SSH
    check_port("localhost", 22)  # SSH
    
    # Test database ports
    check_port("localhost", 5432)  # PostgreSQL
    check_port("localhost", 3306)  # MySQL
    check_port("localhost", 27017)  # MongoDB
    
    # Test binding to port 8080
    test_binding(8080)

def check_stdout_properties():
    """Check properties of stdin/stdout."""
    print_section("STDIN/STDOUT PROPERTIES")
    
    print(f"stdin isatty: {sys.stdin.isatty()}")
    print(f"stdout isatty: {sys.stdout.isatty()}")
    print(f"stderr isatty: {sys.stderr.isatty()}")
    print(f"stdin/stdout encoding: {sys.stdin.encoding}/{sys.stdout.encoding}")

def main():
    print_section("SYSTEM INFORMATION")
    print(f"OS: {platform.system()} {platform.release()} {platform.version()}")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Hostname: {socket.gethostname()}")
    
    try:
        ip_address = socket.gethostbyname(socket.gethostname())
        print(f"IP Address: {ip_address}")
    except Exception as e:
        print(f"Failed to get IP address: {e}")
    
    get_network_interfaces()
    test_socket_connections()
    check_running_processes()
    check_internet_connectivity()
    check_dns_resolution()
    check_open_ports()
    check_stdout_properties()
    
    print_section("DIAGNOSTIC INFORMATION GATHERING COMPLETED")
    print("Please check the output above for any errors or issues.")

if __name__ == "__main__":
    main()
