# REMINDER - Video Quality Regression

## Issue
The original video I recorded 5 months ago worked much better than current results. Need to revert back to that state and check what changed... sigh.

## Action Items
- [ ] Find git commit from ~5 months ago that produced the good video
- [ ] Compare that state with current codebase 
- [ ] Identify key changes that may have degraded video quality
- [ ] Selectively revert problematic changes while keeping performance improvements

## Notes
- RMS method changes didn't work better than local spectral max
- Window time set to 10 seconds
- May need to go back further in git history to find the sweet spot