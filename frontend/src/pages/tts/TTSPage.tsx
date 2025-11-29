/**
 * TTS (Text-to-Speech) Page
 * 
 * Convert text content to audio in multiple Indian languages
 */

import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Volume2,
  Play,
  Pause,
  Download,
  Languages,
  Sliders,
  FileText,
  Mic,
  SkipBack,
  SkipForward,
  Volume1,
  VolumeX
} from 'lucide-react';
import { Button } from '../../components/ui/Button/Button';
import { Spinner } from '../../components/ui/Spinner/Spinner';
import { pageVariants, staggerItem, staggerContainer } from '../../lib/animations';
import { apiClient } from '../../services/client';

// Voice options
const VOICES = [
  { id: 'female_1', name: 'Priya (Female)', language: 'hi' },
  { id: 'male_1', name: 'Rahul (Male)', language: 'hi' },
  { id: 'female_2', name: 'Anita (Female)', language: 'hi' },
  { id: 'male_2', name: 'Vijay (Male)', language: 'hi' },
];

const LANGUAGES = [
  { code: 'hi', name: 'Hindi', nativeName: 'हिंदी' },
  { code: 'en', name: 'English', nativeName: 'English' },
  { code: 'bn', name: 'Bengali', nativeName: 'বাংলা' },
  { code: 'te', name: 'Telugu', nativeName: 'తెలుగు' },
  { code: 'ta', name: 'Tamil', nativeName: 'தமிழ்' },
  { code: 'mr', name: 'Marathi', nativeName: 'मराठी' },
  { code: 'gu', name: 'Gujarati', nativeName: 'ગુજરાતી' },
  { code: 'kn', name: 'Kannada', nativeName: 'ಕನ್ನಡ' },
  { code: 'ml', name: 'Malayalam', nativeName: 'മലയാളം' },
  { code: 'pa', name: 'Punjabi', nativeName: 'ਪੰਜਾਬੀ' },
  { code: 'or', name: 'Odia', nativeName: 'ଓଡ଼ିଆ' },
  { code: 'as', name: 'Assamese', nativeName: 'অসমীয়া' },
];

interface TTSResult {
  audio_url: string;
  duration: number;
  file_size: number;
  language: string;
  voice: string;
}

export function TTSPage() {
  const [text, setText] = useState('');
  const [language, setLanguage] = useState('hi');
  const [voice, setVoice] = useState('female_1');
  const [speed, setSpeed] = useState(1.0);
  const [pitch, setPitch] = useState(1.0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState<TTSResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Audio player state
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);

  // Update audio time
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateTime = () => setCurrentTime(audio.currentTime);
    const updateDuration = () => setDuration(audio.duration);
    const onEnded = () => setIsPlaying(false);

    audio.addEventListener('timeupdate', updateTime);
    audio.addEventListener('loadedmetadata', updateDuration);
    audio.addEventListener('ended', onEnded);

    return () => {
      audio.removeEventListener('timeupdate', updateTime);
      audio.removeEventListener('loadedmetadata', updateDuration);
      audio.removeEventListener('ended', onEnded);
    };
  }, [result]);

  const handleGenerate = async () => {
    if (!text.trim()) {
      setError('Please enter some text to convert');
      return;
    }

    setIsGenerating(true);
    setError(null);

    try {
      const response = await apiClient.post('/api/v1/content/tts', {
        text: text.trim(),
        language,
        voice,
        speed,
        pitch,
      });

      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate audio');
    } finally {
      setIsGenerating(false);
    }
  };

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio) return;

    const time = parseFloat(e.target.value);
    audio.currentTime = time;
    setCurrentTime(time);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio) return;

    const vol = parseFloat(e.target.value);
    audio.volume = vol;
    setVolume(vol);
  };

  const skip = (seconds: number) => {
    const audio = audioRef.current;
    if (!audio) return;

    audio.currentTime = Math.min(Math.max(audio.currentTime + seconds, 0), duration);
  };

  const formatTime = (time: number): string => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="container py-8 max-w-6xl mx-auto"
    >
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
            <Volume2 className="w-6 h-6 text-white" />
          </div>
          Text to Speech
        </h1>
        <p className="text-muted-foreground mt-2">
          Convert educational content to audio in multiple Indian languages
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column - Input */}
        <div className="lg:col-span-2 space-y-6">
          {/* Text Input */}
          <motion.div 
            variants={staggerItem}
            className="bg-card rounded-xl border border-border p-6"
          >
            <label className="block text-sm font-medium text-foreground mb-3">
              <FileText className="w-4 h-4 inline mr-2" />
              Enter Text
            </label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste or type your educational content here..."
              className="w-full h-64 px-4 py-3 rounded-lg border border-input bg-background text-foreground placeholder:text-muted-foreground resize-none focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <div className="flex justify-between mt-3 text-sm text-muted-foreground">
              <span>{text.length} characters</span>
              <span>~{Math.ceil(text.split(' ').filter(Boolean).length / 150)} min audio</span>
            </div>
          </motion.div>

          {/* Audio Player */}
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl p-6 text-white"
            >
              <audio ref={audioRef} src={result.audio_url} preload="metadata" />
              
              <div className="flex items-center justify-between mb-4">
                <div>
                  <p className="font-semibold">Generated Audio</p>
                  <p className="text-sm text-white/70">
                    {LANGUAGES.find(l => l.code === result.language)?.name} • {VOICES.find(v => v.id === result.voice)?.name}
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  className="bg-white/20 border-white/30 text-white hover:bg-white/30"
                  onClick={() => window.open(result.audio_url, '_blank')}
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download
                </Button>
              </div>

              {/* Progress bar */}
              <div className="mb-4">
                <input
                  type="range"
                  min={0}
                  max={duration || 100}
                  value={currentTime}
                  onChange={handleSeek}
                  className="w-full h-2 bg-white/30 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white"
                />
                <div className="flex justify-between text-xs text-white/70 mt-1">
                  <span>{formatTime(currentTime)}</span>
                  <span>{formatTime(duration)}</span>
                </div>
              </div>

              {/* Controls */}
              <div className="flex items-center justify-center gap-4">
                <button 
                  onClick={() => skip(-10)}
                  className="p-2 hover:bg-white/20 rounded-full transition-colors"
                >
                  <SkipBack className="w-5 h-5" />
                </button>
                <button 
                  onClick={togglePlay}
                  className="w-14 h-14 bg-white rounded-full flex items-center justify-center shadow-lg hover:scale-105 transition-transform"
                >
                  {isPlaying ? (
                    <Pause className="w-6 h-6 text-purple-600" />
                  ) : (
                    <Play className="w-6 h-6 text-purple-600 ml-1" />
                  )}
                </button>
                <button 
                  onClick={() => skip(10)}
                  className="p-2 hover:bg-white/20 rounded-full transition-colors"
                >
                  <SkipForward className="w-5 h-5" />
                </button>
              </div>

              {/* Volume */}
              <div className="flex items-center gap-3 mt-4">
                {volume === 0 ? (
                  <VolumeX className="w-5 h-5" />
                ) : volume < 0.5 ? (
                  <Volume1 className="w-5 h-5" />
                ) : (
                  <Volume2 className="w-5 h-5" />
                )}
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.1}
                  value={volume}
                  onChange={handleVolumeChange}
                  className="flex-1 h-2 bg-white/30 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white"
                />
              </div>
            </motion.div>
          )}

          {/* Error */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-4 bg-error/10 border border-error/30 rounded-lg text-error"
            >
              {error}
            </motion.div>
          )}
        </div>

        {/* Right Column - Settings */}
        <motion.div variants={staggerContainer} className="space-y-6">
          {/* Language */}
          <motion.div 
            variants={staggerItem}
            className="bg-card rounded-xl border border-border p-6"
          >
            <label className="block text-sm font-medium text-foreground mb-3">
              <Languages className="w-4 h-4 inline mr-2" />
              Language
            </label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full px-4 py-3 rounded-lg border border-input bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            >
              {LANGUAGES.map(lang => (
                <option key={lang.code} value={lang.code}>
                  {lang.nativeName} ({lang.name})
                </option>
              ))}
            </select>
          </motion.div>

          {/* Voice */}
          <motion.div 
            variants={staggerItem}
            className="bg-card rounded-xl border border-border p-6"
          >
            <label className="block text-sm font-medium text-foreground mb-3">
              <Mic className="w-4 h-4 inline mr-2" />
              Voice
            </label>
            <div className="space-y-2">
              {VOICES.map(v => (
                <button
                  key={v.id}
                  onClick={() => setVoice(v.id)}
                  className={`w-full px-4 py-3 rounded-lg border text-left transition-all ${
                    voice === v.id
                      ? 'border-primary bg-primary/10 text-primary'
                      : 'border-input bg-background text-foreground hover:border-primary/50'
                  }`}
                >
                  {v.name}
                </button>
              ))}
            </div>
          </motion.div>

          {/* Speed & Pitch */}
          <motion.div 
            variants={staggerItem}
            className="bg-card rounded-xl border border-border p-6"
          >
            <label className="block text-sm font-medium text-foreground mb-4">
              <Sliders className="w-4 h-4 inline mr-2" />
              Audio Settings
            </label>
            
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-muted-foreground">Speed</span>
                  <span className="font-medium">{speed.toFixed(1)}x</span>
                </div>
                <input
                  type="range"
                  min={0.5}
                  max={2}
                  step={0.1}
                  value={speed}
                  onChange={(e) => setSpeed(parseFloat(e.target.value))}
                  className="w-full h-2 bg-muted rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary"
                />
              </div>

              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-muted-foreground">Pitch</span>
                  <span className="font-medium">{pitch.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min={0.5}
                  max={2}
                  step={0.1}
                  value={pitch}
                  onChange={(e) => setPitch(parseFloat(e.target.value))}
                  className="w-full h-2 bg-muted rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary"
                />
              </div>
            </div>
          </motion.div>

          {/* Generate Button */}
          <Button
            onClick={handleGenerate}
            disabled={isGenerating || !text.trim()}
            className="w-full py-4 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600"
          >
            {isGenerating ? (
              <>
                <Spinner size="sm" className="mr-2" />
                Generating Audio...
              </>
            ) : (
              <>
                <Volume2 className="w-5 h-5 mr-2" />
                Generate Audio
              </>
            )}
          </Button>
        </motion.div>
      </div>
    </motion.div>
  );
}

export default TTSPage;
