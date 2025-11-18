"""
Retry utilities with exponential backoff.

Provides decorator and helper functions for implementing retry logic
with exponential backoff across the OCR pipeline.
"""

from typing import Callable, Any
import time


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0
) -> Callable:
    """
    Decorator for retrying a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay between retries
        max_delay: Maximum delay cap in seconds
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        print(f"⚠️  Attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}")
                        print(f"   Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        print(f"❌ All {max_retries + 1} attempts failed")
            
            raise last_exception
        
        return wrapper
    return decorator
