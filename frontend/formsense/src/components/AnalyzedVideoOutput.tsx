
import React from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { PlayCircle } from 'lucide-react';

interface AnalyzedVideoOutputProps {
  isVisible: boolean;
  videoFile: File | null;
}

const AnalyzedVideoOutput: React.FC<AnalyzedVideoOutputProps> = ({ isVisible, videoFile }) => {
  if (!isVisible) {
    return (
      <Card className="p-6">
        <div className="text-center text-muted-foreground">
          <div className="text-4xl mb-4">🎬</div>
          <h3 className="text-lg font-semibold mb-2">Analyzed Video Output</h3>
          <p>Your analyzed workout video will appear here</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-6 animate-fade-in">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Analyzed Video Output</h3>
          <Badge variant="default" className="bg-fitness-success">
            Analysis Complete
          </Badge>
        </div>

        <div className="relative">
          {videoFile && (
            <video
              src={URL.createObjectURL(videoFile)}
              controls
              className="w-full rounded-lg"
              style={{ maxHeight: '400px' }}
            />
          )}
          
          {/* Overlay showing pose estimation visualization */}
          <div className="absolute inset-0 pointer-events-none rounded-lg">
            <div className="relative w-full h-full">
              {/* Simulated pose points - in real implementation, these would be dynamically positioned */}
              <div className="absolute top-1/4 left-1/2 w-2 h-2 bg-fitness-accent rounded-full animate-pulse-gentle transform -translate-x-1/2" />
              <div className="absolute top-1/3 left-1/2 w-2 h-2 bg-fitness-accent rounded-full animate-pulse-gentle transform -translate-x-1/2" />
              <div className="absolute top-1/2 left-1/2 w-2 h-2 bg-fitness-accent rounded-full animate-pulse-gentle transform -translate-x-1/2" />
              <div className="absolute top-2/3 left-1/2 w-2 h-2 bg-fitness-accent rounded-full animate-pulse-gentle transform -translate-x-1/2" />
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-2 text-sm text-muted-foreground">
          <PlayCircle className="h-4 w-4" />
          <span>Video includes pose estimation overlay and form analysis</span>
        </div>
      </div>
    </Card>
  );
};

export default AnalyzedVideoOutput;
