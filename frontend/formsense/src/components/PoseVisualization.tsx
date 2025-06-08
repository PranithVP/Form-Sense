import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';

interface PoseVisualizationProps {
  isAnalyzing: boolean;
  videoFile: File | null;
  processedVideoUrl?: string;
}

const PoseVisualization: React.FC<PoseVisualizationProps> = ({ 
  isAnalyzing, 
  videoFile,
  processedVideoUrl 
}) => {
  const [progress, setProgress] = useState(0);
  const [analysisStage, setAnalysisStage] = useState('');

  useEffect(() => {
    if (isAnalyzing) {
      const stages = [
        'Loading video...',
        'Detecting pose landmarks...',
        'Analyzing form...',
        'Calculating scores...',
        'Generating feedback...'
      ];
      
      let currentStage = 0;
      const interval = setInterval(() => {
        setProgress((prev) => {
          const newProgress = prev + 20;
          if (newProgress <= 100) {
            setAnalysisStage(stages[Math.floor(newProgress / 20) - 1] || stages[stages.length - 1]);
            return newProgress;
          }
          clearInterval(interval);
          return 100;
        });
      }, 1000);

      return () => clearInterval(interval);
    } else {
      setProgress(0);
      setAnalysisStage('');
    }
  }, [isAnalyzing]);

  useEffect(() => {
    if (processedVideoUrl) {
      console.log("[PoseVisualization] Processed video URL received:", processedVideoUrl);
    }
  }, [processedVideoUrl]);

  const handleVideoError = (event: React.SyntheticEvent<HTMLVideoElement, Event>) => {
    console.error("[PoseVisualization] Video playback error:", event.currentTarget.error);
    if (event.currentTarget.error) {
      switch (event.currentTarget.error.code) {
        case event.currentTarget.error.MEDIA_ERR_ABORTED:
          console.error("Video playback aborted.");
          break;
        case event.currentTarget.error.MEDIA_ERR_NETWORK:
          console.error("Video download failed due to network error.");
          break;
        case event.currentTarget.error.MEDIA_ERR_DECODE:
          console.error("Video playback failed due to a decoding error.");
          break;
        case event.currentTarget.error.MEDIA_ERR_SRC_NOT_SUPPORTED:
          console.error("Video format not supported or invalid source.");
          break;
        default:
          console.error("An unknown video error occurred.");
          break;
      }
    }
  };

  return (
    <Card className="p-6">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Pose Analysis</h3>
          <Badge variant={isAnalyzing ? "default" : "secondary"}>
            {isAnalyzing ? "Analyzing" : "Ready"}
          </Badge>
        </div>

        {processedVideoUrl && !isAnalyzing && (
          <div className="relative">
            <video
              src={processedVideoUrl}
              className="w-full rounded-lg"
              style={{ maxHeight: '400px' }}
              controls
              autoPlay
              onError={handleVideoError}
            />
          </div>
        )}

        {videoFile && !processedVideoUrl && !isAnalyzing && (
          <div className="relative">
            <video
              src={URL.createObjectURL(videoFile)}
              className="w-full rounded-lg"
              style={{ maxHeight: '400px' }}
              poster="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect width='100' height='100' fill='%23f3f4f6'/%3E%3C/svg%3E"
            />
            <div className="absolute inset-0 bg-black/50 rounded-lg flex items-center justify-center">
              <div className="text-center text-white">
                <div className="animate-pulse-gentle mb-2">🎯</div>
                <p className="text-sm">Analysis will appear here</p>
              </div>
            </div>
          </div>
        )}

        {isAnalyzing && (
          <div className="space-y-4">
            <div className="relative">
              <div className="w-full h-64 bg-muted rounded-lg flex items-center justify-center">
                <div className="text-center">
                  <div className="animate-pulse-gentle text-4xl mb-4">🔄</div>
                  <p className="text-lg font-medium">{analysisStage}</p>
                </div>
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Analysis Progress</span>
                <span>{progress}%</span>
              </div>
              <Progress value={progress} className="w-full" />
            </div>
          </div>
        )}

        {!videoFile && !isAnalyzing && (
          <div className="h-64 bg-muted rounded-lg flex items-center justify-center">
            <div className="text-center text-muted-foreground">
              <div className="text-4xl mb-4">📹</div>
              <p>Upload a video to start analysis</p>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default PoseVisualization;
