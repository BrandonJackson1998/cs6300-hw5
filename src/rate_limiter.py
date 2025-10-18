"""
Rate Limiter for Gemini API

This module provides a RateLimiter class to manage Gemini API quota
and prevent exceeding daily/hourly limits by tracking requests and
enforcing delays when approaching rate limits.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QuotaConfig:
    """Configuration for API quota limits."""
    requests_per_minute: int = 15  # Gemini API free tier limit
    requests_per_day: int = 1500   # Gemini API free tier daily limit
    tokens_per_minute: int = 32000  # Token limit per minute
    tokens_per_day: int = 50000     # Token limit per day
    
    # Safety margins (percentage of actual limit to use)
    safety_margin: float = 0.9


class RateLimiter:
    """
    Manages Gemini API quota to prevent exceeding daily/hourly limits.
    Tracks requests and enforces delays when approaching rate limits.
    
    This class provides thread-safe rate limiting for the Gemini API by:
    - Tracking requests per minute and per day
    - Tracking token usage per minute and per day
    - Enforcing delays when approaching limits
    - Providing safety margins to avoid hitting hard limits
    
    Example:
        >>> rate_limiter = RateLimiter()
        >>> # Before making an API call
        >>> rate_limiter.wait_if_needed(estimated_tokens=1000)
        >>> # Make your API call here
        >>> response = gemini_api_call()
        >>> # Record the actual usage
        >>> rate_limiter.record_request(actual_tokens=800)
    """
    
    def __init__(self, config: Optional[QuotaConfig] = None):
        """
        Initialize the rate limiter.
        
        Args:
            config: Optional QuotaConfig object. If None, uses default limits.
        """
        self.config = config or QuotaConfig()
        self._lock = threading.Lock()
        
        # Request tracking
        self._minute_requests: Dict[datetime, int] = {}
        self._daily_requests: Dict[datetime, int] = {}
        
        # Token tracking
        self._minute_tokens: Dict[datetime, int] = {}
        self._daily_tokens: Dict[datetime, int] = {}
        
        # Last request time for minimum delay enforcement
        self._last_request_time: Optional[float] = None
        
        # Minimum delay between requests (in seconds)
        self._min_delay = 60.0 / (self.config.requests_per_minute * self.config.safety_margin)
        
        logger.info(f"RateLimiter initialized with config: {self.config}")
    
    def wait_if_needed(self, estimated_tokens: int = 1000) -> None:
        """
        Wait if necessary to respect rate limits before making a request.
        
        Args:
            estimated_tokens: Estimated number of tokens for the upcoming request
        """
        with self._lock:
            current_time = time.time()
            
            # Clean up old entries
            self._cleanup_old_entries()
            
            # Check if we need to wait based on request limits
            self._wait_for_request_limit()
            
            # Check if we need to wait based on token limits
            self._wait_for_token_limit(estimated_tokens)
            
            # Enforce minimum delay between requests
            self._enforce_minimum_delay(current_time)
    
    def record_request(self, actual_tokens: int = 0) -> None:
        """
        Record a completed request and its token usage.
        
        Args:
            actual_tokens: Actual number of tokens used in the request
        """
        with self._lock:
            now = datetime.now()
            current_minute = now.replace(second=0, microsecond=0)
            current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Record request
            self._minute_requests[current_minute] = self._minute_requests.get(current_minute, 0) + 1
            self._daily_requests[current_day] = self._daily_requests.get(current_day, 0) + 1
            
            # Record tokens
            if actual_tokens > 0:
                self._minute_tokens[current_minute] = self._minute_tokens.get(current_minute, 0) + actual_tokens
                self._daily_tokens[current_day] = self._daily_tokens.get(current_day, 0) + actual_tokens
            
            # Update last request time
            self._last_request_time = time.time()
            
            logger.debug(f"Recorded request: {actual_tokens} tokens")
    
    def get_current_usage(self) -> Dict[str, Any]:
        """
        Get current usage statistics.
        
        Returns:
            Dictionary containing current usage statistics
        """
        with self._lock:
            self._cleanup_old_entries()
            
            now = datetime.now()
            current_minute = now.replace(second=0, microsecond=0)
            current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Count requests in current minute and day
            minute_requests = sum(
                count for timestamp, count in self._minute_requests.items()
                if timestamp >= current_minute - timedelta(minutes=1)
            )
            
            daily_requests = self._daily_requests.get(current_day, 0)
            
            # Count tokens in current minute and day
            minute_tokens = sum(
                count for timestamp, count in self._minute_tokens.items()
                if timestamp >= current_minute - timedelta(minutes=1)
            )
            
            daily_tokens = self._daily_tokens.get(current_day, 0)
            
            return {
                'requests_per_minute': minute_requests,
                'requests_per_day': daily_requests,
                'tokens_per_minute': minute_tokens,
                'tokens_per_day': daily_tokens,
                'max_requests_per_minute': int(self.config.requests_per_minute * self.config.safety_margin),
                'max_requests_per_day': int(self.config.requests_per_day * self.config.safety_margin),
                'max_tokens_per_minute': int(self.config.tokens_per_minute * self.config.safety_margin),
                'max_tokens_per_day': int(self.config.tokens_per_day * self.config.safety_margin),
            }
    
    def _cleanup_old_entries(self) -> None:
        """Remove old tracking entries to prevent memory leaks."""
        now = datetime.now()
        cutoff_minute = now - timedelta(minutes=2)
        cutoff_day = now - timedelta(days=2)
        
        # Clean up minute tracking
        self._minute_requests = {
            ts: count for ts, count in self._minute_requests.items()
            if ts >= cutoff_minute
        }
        self._minute_tokens = {
            ts: count for ts, count in self._minute_tokens.items()
            if ts >= cutoff_minute
        }
        
        # Clean up daily tracking
        self._daily_requests = {
            ts: count for ts, count in self._daily_requests.items()
            if ts >= cutoff_day
        }
        self._daily_tokens = {
            ts: count for ts, count in self._daily_tokens.items()
            if ts >= cutoff_day
        }
    
    def _wait_for_request_limit(self) -> None:
        """Wait if we're approaching request limits."""
        now = datetime.now()
        current_minute = now.replace(second=0, microsecond=0)
        current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Check minute limit
        minute_requests = sum(
            count for timestamp, count in self._minute_requests.items()
            if timestamp >= current_minute - timedelta(minutes=1)
        )
        
        max_minute_requests = int(self.config.requests_per_minute * self.config.safety_margin)
        if minute_requests >= max_minute_requests:
            wait_time = 60 - now.second
            logger.warning(f"Request rate limit approaching. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
        
        # Check daily limit
        daily_requests = self._daily_requests.get(current_day, 0)
        max_daily_requests = int(self.config.requests_per_day * self.config.safety_margin)
        if daily_requests >= max_daily_requests:
            tomorrow = current_day + timedelta(days=1)
            wait_time = (tomorrow - now).total_seconds()
            logger.error(f"Daily request limit reached. Waiting until tomorrow ({wait_time/3600:.1f} hours)...")
            time.sleep(wait_time)
    
    def _wait_for_token_limit(self, estimated_tokens: int) -> None:
        """Wait if adding estimated tokens would exceed limits."""
        now = datetime.now()
        current_minute = now.replace(second=0, microsecond=0)
        current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Check minute token limit
        minute_tokens = sum(
            count for timestamp, count in self._minute_tokens.items()
            if timestamp >= current_minute - timedelta(minutes=1)
        )
        
        max_minute_tokens = int(self.config.tokens_per_minute * self.config.safety_margin)
        if minute_tokens + estimated_tokens > max_minute_tokens:
            wait_time = 60 - now.second
            logger.warning(f"Token rate limit approaching. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
        
        # Check daily token limit
        daily_tokens = self._daily_tokens.get(current_day, 0)
        max_daily_tokens = int(self.config.tokens_per_day * self.config.safety_margin)
        if daily_tokens + estimated_tokens > max_daily_tokens:
            tomorrow = current_day + timedelta(days=1)
            wait_time = (tomorrow - now).total_seconds()
            logger.error(f"Daily token limit would be exceeded. Waiting until tomorrow ({wait_time/3600:.1f} hours)...")
            time.sleep(wait_time)
    
    def _enforce_minimum_delay(self, current_time: float) -> None:
        """Enforce minimum delay between requests."""
        if self._last_request_time is not None:
            time_since_last = current_time - self._last_request_time
            if time_since_last < self._min_delay:
                wait_time = self._min_delay - time_since_last
                logger.debug(f"Enforcing minimum delay: waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
    
    def reset_daily_limits(self) -> None:
        """Reset daily tracking (useful for testing or manual reset)."""
        with self._lock:
            now = datetime.now()
            current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            self._daily_requests.pop(current_day, None)
            self._daily_tokens.pop(current_day, None)
            logger.info("Daily limits reset")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass