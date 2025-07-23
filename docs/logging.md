# Vehicle Router Logging System

The Vehicle Router application includes a comprehensive logging system that provides detailed insights into application performance, user interactions, and system behavior.

## 📁 Log File Organization

The logging system creates separate directories for different components:

```
logs/
├── main/                    # CLI application logs
│   ├── main_YYYYMMDD_HHMMSS.log    # Timestamped log files
│   └── latest.log          # Symlink to most recent log
└── app/                     # Streamlit web application logs
    ├── app_SESSION_YYYYMMDD_HHMMSS.log    # Session-specific log files
    └── session_SESSION_latest.log          # Symlink to session's latest log
```

## 🚀 Features

### **Enhanced Formatting**
- **Timestamps**: All logs include precise timestamps
- **Color Coding**: Console output uses colors for different log levels
- **Session Tracking**: Web app logs include unique session identifiers
- **Performance Metrics**: Built-in timing and memory usage tracking

### **Automatic Rotation**
- **File Size Limits**: Log files are automatically rotated when they exceed size limits
  - Main application: 10MB per file, 5 backup files
  - Web application: 5MB per file, 3 backup files
- **Cleanup**: Old log files are automatically cleaned up to prevent disk space issues

### **Performance Tracking**
- **Operation Timing**: Track execution time for key operations
- **Memory Usage**: Monitor memory consumption at critical points
- **System Information**: Automatic logging of system specifications

## 🖥️ Main Application Logging

The CLI application (`src/main.py`) uses enhanced logging with performance tracking:

### **Log Levels**
- **INFO**: General operation information, progress updates
- **WARNING**: Non-critical issues that don't stop execution
- **ERROR**: Critical errors that may cause failures
- **DEBUG**: Detailed debugging information (file only)

### **Performance Tracking**
```python
# Automatic timing of major operations
logger.perf.start_timer("Data Generation")
# ... operation code ...
duration = logger.perf.end_timer("Data Generation")

# Memory usage tracking
logger.perf.log_memory_usage("After optimization")
```

### **Example Log Output**
```
[INFO] 2025-07-23 11:59:03 - vehicle_router.main - 🚛 VEHICLE ROUTER - MAIN APPLICATION LOGGING INITIALIZED
[INFO] 2025-07-23 11:59:03 - vehicle_router.main - 📅 Session started: 2025-07-23 11:59:03
[INFO] 2025-07-23 11:59:03 - vehicle_router.main - ⏱️  Started: Complete Workflow
[INFO] 2025-07-23 11:59:04 - vehicle_router.main - ✅ Completed: Data Generation (Duration: 0.85s)
[INFO] 2025-07-23 11:59:05 - vehicle_router.main - ✅ Completed: Optimization (Duration: 1.23s)
[INFO] 2025-07-23 11:59:05 - vehicle_router.main - ✅ WORKFLOW COMPLETED SUCCESSFULLY IN 2.08 SECONDS
```

## 🌐 Streamlit Application Logging

The web application (`app/streamlit_app.py`) uses session-based logging:

### **Session Management**
- **Unique Session IDs**: Each user session gets a unique 8-character identifier
- **Session-Specific Logs**: All user interactions are logged with session context
- **Performance Tracking**: Web app operations are timed and logged

### **User Interaction Logging**
- **Data Loading**: Track when users load example or custom data
- **Optimization Runs**: Log optimization parameters and results
- **Distance Method Changes**: Track switches between simulated and real distances
- **Documentation Access**: Log which documentation sections users view

### **Example Log Output**
```
[INFO] 11:59:04 - Session:abc12345 - 🌐 VEHICLE ROUTER - STREAMLIT APP LOGGING INITIALIZED
[INFO] 11:59:04 - Session:abc12345 - 📊 User initiated example data loading
[INFO] 11:59:05 - Session:abc12345 - ✅ Example data loaded successfully in 0.45s
[INFO] 11:59:06 - Session:abc12345 - 🚀 User initiated optimization
[INFO] 11:59:06 - Session:abc12345 -    Method: standard
[INFO] 11:59:07 - Session:abc12345 - ✅ Optimization completed successfully in 1.2s
```

## 📊 Log Content Examples

### **System Information** (logged at startup)
```
[INFO] 🖥️  SYSTEM INFORMATION:
[INFO]    Platform: macOS-14.5-arm64-arm-64bit
[INFO]    Python version: 3.11.7
[INFO]    Architecture: 64bit
[INFO]    CPU cores: 8
[INFO]    Memory: 16.0 GB
```

### **Optimization Configuration**
```
[INFO] 🚀 OPTIMIZATION STARTING:
[INFO]    Method: enhanced
[INFO]    Depot: 08025
[INFO]    Real distances: True
[INFO]    Timeout: 120s
[INFO]    Weights: Cost=0.60, Distance=0.40
```

### **Performance Metrics**
```
[INFO] ⏱️  Started: Complete Workflow
[INFO] ✅ Completed: Data Generation (Duration: 0.85s)
[INFO] ✅ Completed: Optimization (Duration: 12.34s)
[INFO] 🧠 Memory usage - After optimization: 245.3 MB
[INFO] ✅ Completed: Complete Workflow (Duration: 15.67s)
```

### **Error Handling**
```
[ERROR] ❌ Optimization failed
[ERROR] 💥 CRITICAL ERROR in workflow:
[ERROR]    Error: Invalid depot location: 99999
[ERROR]    Full traceback:
[ERROR] Traceback (most recent call last):
[ERROR]   File "src/main.py", line 95, in run
```

## 🔧 Configuration

### **Log Levels**
Control verbosity with command-line arguments:
```bash
# Standard logging
python src/main.py

# Quiet mode (warnings and errors only)
python src/main.py --quiet

# Debug mode (all messages)
python src/main.py --log-level DEBUG
```

### **File Management**
- **Automatic Cleanup**: Old logs are cleaned up automatically (configurable, default 7 days)
- **Manual Cleanup**: Run cleanup script to remove old logs
- **Symlinks**: Latest logs are always accessible via `latest.log` symlinks

## 📈 Monitoring and Analysis

### **Performance Analysis**
Use the logs to analyze:
- **Execution Times**: Compare optimization times across different methods
- **Memory Usage**: Monitor memory consumption for large datasets
- **User Patterns**: Understand how users interact with the web interface
- **Error Patterns**: Identify common failure points

### **Log Aggregation**
For production deployments:
- Logs are structured and easily parseable
- JSON format can be enabled for log aggregation tools
- Session-based tracking enables user journey analysis

### **Example Analysis Queries**
```bash
# Find all optimization runs that took longer than 30 seconds
grep "Completed: Complete" logs/main/*.log | grep -E "\([3-9][0-9]\.[0-9]+s\)"

# Count optimization methods used
grep "Method:" logs/app/*.log | cut -d: -f4 | sort | uniq -c

# Find all errors in the last 24 hours
find logs/ -name "*.log" -newerct "24 hours ago" -exec grep "ERROR" {} +
```

## 🛠️ Development and Debugging

### **Debug Mode**
Enable detailed logging for troubleshooting:
```python
from vehicle_router.logger_config import setup_main_logging

logger = setup_main_logging(
    log_level="DEBUG",
    log_to_console=True,
    enable_performance_tracking=True
)
```

### **Custom Logging**
Add application-specific logging:
```python
# In your modules
import logging
logger = logging.getLogger(__name__)

logger.info("📋 Processing order batch")
logger.warning("⚠️ Capacity constraint is tight")
logger.error("❌ Invalid postal code format")
```

### **Performance Profiling**
Use the built-in performance tracking:
```python
# Start timing
logger.perf.start_timer("Custom Operation")

# Your code here
time.sleep(1)

# End timing and log duration
duration = logger.perf.end_timer("Custom Operation")

# Log memory usage
logger.perf.log_memory_usage("After heavy computation")
```

## 🔒 Privacy and Security

### **Data Protection**
- **No Sensitive Data**: Logs contain operational data only, no customer information
- **Configurable Retention**: Automatic cleanup prevents indefinite data storage
- **Local Storage**: All logs are stored locally, no external transmission

### **Access Control**
- **File Permissions**: Log files respect system file permissions
- **Session Isolation**: Web app sessions are logged separately
- **Audit Trail**: All user actions in web interface are logged for audit purposes

## 📚 Further Reading

- **Optimization Methods**: See `docs/methods.md` for algorithm details
- **Usage Examples**: See `docs/usage.md` for complete usage guide
- **System Requirements**: See main `README.md` for installation and setup 