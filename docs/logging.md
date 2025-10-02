# Vehicle Router Logging System

The Vehicle Router includes comprehensive logging for both CLI and web applications.

## Log File Organization

```
logs/
├── main/                    # CLI application logs
│   ├── main_YYYYMMDD_HHMMSS.log
│   └── latest.log
└── app/                     # Streamlit web application logs
    ├── app_SESSION_YYYYMMDD_HHMMSS.log
    └── session_SESSION_latest.log
```

## Features

- **Timestamps**: All logs include precise timestamps
- **Color Coding**: Console output uses colors for different log levels
- **Session Tracking**: Web app logs include unique session identifiers
- **Performance Metrics**: Built-in timing and memory usage tracking
- **Automatic Rotation**: Log files are rotated when they exceed size limits

## Log Levels

- **INFO**: General operation information, progress updates
- **WARNING**: Non-critical issues that don't stop execution
- **ERROR**: Critical errors that may cause failures
- **DEBUG**: Detailed debugging information (file only)

## Performance Tracking

The system automatically tracks:
- Operation timing for key processes
- Memory usage at critical points
- System information and specifications
- Optimization performance metrics

## Configuration

Logging can be configured through:
- Environment variables
- Configuration files
- Programmatic setup

Default settings:
- Main application: 10MB per file, 5 backup files
- Web application: 5MB per file, 3 backup files