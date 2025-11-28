import { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import { cn } from '../ui/utils';
import { IconButton } from '../ui/IconButton';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { Tooltip } from '../ui/Tooltip';

export interface AudioPlayerProps {
  /** Audio source URL */
  src: string;
  /** Audio title */
  title?: string;
  /** Audio language */
  language?: string;
  /** Show download button */
  showDownload?: boolean;
  /** Download filename */
  downloadFilename?: string;
  /** Show speed control */
  showSpeedControl?: boolean;
  /** Auto play on load */
  autoPlay?: boolean;
  /** Callback when playback ends */
  onEnded?: () => void;
  /** Additional class name */
  className?: string;
}

// Icons
const PlayIcon = () => (
  <svg className="w-full h-full" fill="currentColor" viewBox="0 0 24 24">
    <path d="M8 5v14l11-7z" />
  </svg>
);

const PauseIcon = () => (
  <svg className="w-full h-full" fill="currentColor" viewBox="0 0 24 24">
    <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
  </svg>
);

const VolumeIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
  </svg>
);

const VolumeMuteIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2" />
  </svg>
);

const DownloadIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
  </svg>
);

const RewindIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12.066 11.2a1 1 0 000 1.6l5.334 4A1 1 0 0019 16V8a1 1 0 00-1.6-.8l-5.333 4zM4.066 11.2a1 1 0 000 1.6l5.334 4A1 1 0 0011 16V8a1 1 0 00-1.6-.8l-5.334 4z" />
  </svg>
);

const ForwardIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.933 12.8a1 1 0 000-1.6L6.6 7.2A1 1 0 005 8v8a1 1 0 001.6.8l5.333-4zM19.933 12.8a1 1 0 000-1.6l-5.333-4A1 1 0 0013 8v8a1 1 0 001.6.8l5.333-4z" />
  </svg>
);

const PLAYBACK_SPEEDS = [0.5, 0.75, 1, 1.25, 1.5, 2];

/**
 * Custom audio player component with playback controls, volume, speed, and download.
 * 
 * @example
 * <AudioPlayer
 *   src="/api/audio/task123.mp3"
 *   title="Hindi Translation"
 *   language="hi"
 *   showDownload
 *   showSpeedControl
 * />
 */
export function AudioPlayer({
  src,
  title,
  language,
  showDownload = true,
  downloadFilename = 'audio.mp3',
  showSpeedControl = true,
  autoPlay = false,
  onEnded,
  className,
}: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Format time as MM:SS
  const formatTime = useCallback((time: number): string => {
    if (!isFinite(time)) return '0:00';
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }, []);

  // Progress percentage
  const progress = useMemo(() => {
    return duration > 0 ? (currentTime / duration) * 100 : 0;
  }, [currentTime, duration]);

  // Handle play/pause
  const togglePlay = useCallback(() => {
    if (!audioRef.current) return;
    
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play().catch(err => {
        setError('Failed to play audio');
        console.error('Play error:', err);
      });
    }
  }, [isPlaying]);

  // Seek to position
  const handleSeek = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current || !progressRef.current) return;
    
    const rect = progressRef.current.getBoundingClientRect();
    const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    audioRef.current.currentTime = percent * duration;
  }, [duration]);

  // Skip forward/backward
  const skip = useCallback((seconds: number) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = Math.max(0, Math.min(duration, currentTime + seconds));
  }, [currentTime, duration]);

  // Toggle mute
  const toggleMute = useCallback(() => {
    if (!audioRef.current) return;
    audioRef.current.muted = !isMuted;
    setIsMuted(!isMuted);
  }, [isMuted]);

  // Change volume
  const handleVolumeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (!audioRef.current) return;
    const newVolume = parseFloat(e.target.value);
    audioRef.current.volume = newVolume;
    setVolume(newVolume);
    if (newVolume > 0 && isMuted) {
      audioRef.current.muted = false;
      setIsMuted(false);
    }
  }, [isMuted]);

  // Change playback speed
  const cyclePlaybackSpeed = useCallback(() => {
    if (!audioRef.current) return;
    const currentIndex = PLAYBACK_SPEEDS.indexOf(playbackRate);
    const nextIndex = (currentIndex + 1) % PLAYBACK_SPEEDS.length;
    const newRate = PLAYBACK_SPEEDS[nextIndex];
    audioRef.current.playbackRate = newRate;
    setPlaybackRate(newRate);
  }, [playbackRate]);

  // Handle download
  const handleDownload = useCallback(async () => {
    try {
      const response = await fetch(src);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = downloadFilename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download error:', err);
    }
  }, [src, downloadFilename]);

  // Audio event handlers
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleLoadedMetadata = () => {
      setDuration(audio.duration);
      setIsLoading(false);
    };

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
    };

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => {
      setIsPlaying(false);
      onEnded?.();
    };

    const handleError = () => {
      setError('Failed to load audio');
      setIsLoading(false);
    };

    const handleWaiting = () => setIsLoading(true);
    const handleCanPlay = () => setIsLoading(false);

    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('error', handleError);
    audio.addEventListener('waiting', handleWaiting);
    audio.addEventListener('canplay', handleCanPlay);

    return () => {
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('error', handleError);
      audio.removeEventListener('waiting', handleWaiting);
      audio.removeEventListener('canplay', handleCanPlay);
    };
  }, [onEnded]);

  if (error) {
    return (
      <div className={cn(
        'p-4 rounded-lg bg-error-50 dark:bg-error-900/20 border border-error-200 dark:border-error-800',
        className
      )}>
        <p className="text-sm text-error-600 dark:text-error-400">{error}</p>
      </div>
    );
  }

  return (
    <div
      className={cn(
        'bg-surface-50 dark:bg-surface-800 rounded-xl p-4 border border-surface-200 dark:border-surface-700',
        className
      )}
    >
      <audio ref={audioRef} src={src} autoPlay={autoPlay} preload="metadata" />

      {/* Header */}
      {(title || language) && (
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            {title && (
              <span className="text-sm font-medium text-surface-900 dark:text-surface-100">
                {title}
              </span>
            )}
            {language && (
              <Badge variant="primary" size="sm">
                {language.toUpperCase()}
              </Badge>
            )}
          </div>
        </div>
      )}

      {/* Progress bar */}
      <div
        ref={progressRef}
        className="h-2 bg-surface-200 dark:bg-surface-700 rounded-full cursor-pointer mb-3 group"
        onClick={handleSeek}
        role="slider"
        aria-label="Audio progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={progress}
        tabIndex={0}
      >
        <div
          className="h-full bg-primary-500 rounded-full relative transition-all"
          style={{ width: `${progress}%` }}
        >
          <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-primary-500 rounded-full shadow opacity-0 group-hover:opacity-100 transition-opacity" />
        </div>
      </div>

      {/* Time display */}
      <div className="flex justify-between text-xs text-surface-500 mb-3">
        <span>{formatTime(currentTime)}</span>
        <span>{formatTime(duration)}</span>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between">
        {/* Playback controls */}
        <div className="flex items-center gap-1">
          <Tooltip content="Rewind 10s">
            <IconButton
              icon={<RewindIcon />}
              variant="ghost"
              size="sm"
              aria-label="Rewind 10 seconds"
              onClick={() => skip(-10)}
            />
          </Tooltip>

          <button
            onClick={togglePlay}
            disabled={isLoading}
            className={cn(
              'w-12 h-12 rounded-full flex items-center justify-center',
              'bg-primary-500 text-white hover:bg-primary-600',
              'transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
              isLoading && 'opacity-50 cursor-wait'
            )}
            aria-label={isPlaying ? 'Pause' : 'Play'}
          >
            <span className="w-6 h-6">
              {isPlaying ? <PauseIcon /> : <PlayIcon />}
            </span>
          </button>

          <Tooltip content="Forward 10s">
            <IconButton
              icon={<ForwardIcon />}
              variant="ghost"
              size="sm"
              aria-label="Forward 10 seconds"
              onClick={() => skip(10)}
            />
          </Tooltip>
        </div>

        {/* Secondary controls */}
        <div className="flex items-center gap-2">
          {/* Speed control */}
          {showSpeedControl && (
            <Tooltip content="Playback speed">
              <Button
                variant="ghost"
                size="sm"
                onClick={cyclePlaybackSpeed}
                className="text-xs font-mono min-w-[3rem]"
              >
                {playbackRate}x
              </Button>
            </Tooltip>
          )}

          {/* Volume */}
          <div className="flex items-center gap-1 group">
            <Tooltip content={isMuted ? 'Unmute' : 'Mute'}>
              <IconButton
                icon={isMuted ? <VolumeMuteIcon /> : <VolumeIcon />}
                variant="ghost"
                size="sm"
                aria-label={isMuted ? 'Unmute' : 'Mute'}
                onClick={toggleMute}
              />
            </Tooltip>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={isMuted ? 0 : volume}
              onChange={handleVolumeChange}
              className="w-0 group-hover:w-20 transition-all opacity-0 group-hover:opacity-100 accent-primary-500"
              aria-label="Volume"
            />
          </div>

          {/* Download */}
          {showDownload && (
            <Tooltip content="Download audio">
              <IconButton
                icon={<DownloadIcon />}
                variant="ghost"
                size="sm"
                aria-label="Download audio"
                onClick={handleDownload}
              />
            </Tooltip>
          )}
        </div>
      </div>
    </div>
  );
}

AudioPlayer.displayName = 'AudioPlayer';
