import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Play, Activity, Target, BarChart3, Lightbulb, Loader2, RefreshCw, AlertTriangle, CheckCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import VideoUpload from '@/components/VideoUpload';
import PoseVisualization from '@/components/PoseVisualization';

const Index = () => {
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [processedVideoUrl, setProcessedVideoUrl] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [exerciseTips, setExerciseTips] = useState<string | null>(null);

  const getExerciseTips = (exercise: string): string => {
    const tips: { [key: string]: string } = {
      'Barbell Biceps Curl': 'Keep your elbows close to your body and maintain a controlled motion throughout the exercise.',
      'Squat': 'Keep your back straight, chest up, and ensure your knees don\'t go past your toes.',
      'Deadlift': 'Maintain a neutral spine and keep the bar close to your body throughout the movement.',
      'Bench Press': 'Keep your feet flat on the ground and maintain a slight arch in your lower back.',
      'Overhead Press': 'Keep your core tight and maintain a neutral spine throughout the movement.',
      'default': 'Focus on proper form and controlled movements. Consider consulting a fitness professional for personalized guidance.'
    };
    
    return tips[exercise] || tips.default;
  };

  const handleVideoSelect = (file: File) => {
    setSelectedVideo(file);
    setProcessedVideoUrl(null);
    setAnalysisResult(null);
  };

  const handleClearVideo = () => {
    setSelectedVideo(null);
    setProcessedVideoUrl(null);
    setAnalysisResult(null);
  };

  const handleAnalyze = async () => {
    if (!selectedVideo) {
      alert("Please select a video file first.");
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);
    setExerciseTips(null);

    const formData = new FormData();
    formData.append("file", selectedVideo);

    try {
      console.log("[Frontend] Sending video to backend...");
      const response = await fetch("http://localhost:8000/api/v1/process", {
        method: "POST",
        body: formData,
      });

      console.log(`[Frontend] Backend response status: ${response.status}`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Video processing failed.");
      }

      const responseData = await response.json();
      console.log("[Frontend] Received JSON response from backend:", responseData);

      const { video_data, exercise_data } = responseData;

      if (video_data) {
        // Decode base64 video data and create a Blob
        const videoBlob = await fetch(`data:video/mp4;base64,${video_data}`).then(res => res.blob());
        const videoUrl = URL.createObjectURL(videoBlob);
        setProcessedVideoUrl(videoUrl);
        console.log(`[Frontend] Video blob created, URL: ${videoUrl}`);
      } else {
        console.warn("[Frontend] No video_data received in the response.");
        setProcessedVideoUrl(null);
      }

      if (exercise_data) {
        console.log("[Frontend] Extracted exercise_data:", exercise_data);
        setAnalysisResult({
          classified_exercise: exercise_data.classified_exercise,
          average_angles: exercise_data.average_angles,
          feedback: exercise_data.feedback,
          angle_data: exercise_data.angle_data,
          angle_analysis: exercise_data.angle_analysis
        });
        setExerciseTips(getExerciseTips(exercise_data.classified_exercise));
      } else {
        console.warn("[Frontend] No exercise_data received in the response.");
        setAnalysisResult(null);
        setExerciseTips(null);
      }

    } catch (err) {
      console.error("[Frontend] Error processing video:", err);
      alert(`Error processing video: ${err.message}`);
      setProcessedVideoUrl(null);
      setAnalysisResult(null);
      setExerciseTips(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-fitness-background">
      {/* Header */}
      <header className="bg-gradient-to-r from-fitness-surface to-fitness-surface/80 border-b border-fitness-accent/20">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 gradient-fitness rounded-xl flex items-center justify-center shadow-lg">
                <Activity className="h-7 w-7 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-black">FormSense</h1>
                <p className="text-base text-muted-foreground mt-1">AI-powered workout form analysis</p>
              </div>
            </div>
            <div className="hidden md:flex items-center space-x-6">
              <div className="flex items-center space-x-2 text-muted-foreground">
                <Target className="h-5 w-5" />
                <span className="text-sm">Real-time Analysis</span>
              </div>
              <div className="flex items-center space-x-2 text-muted-foreground">
                <Activity className="h-5 w-5" />
                <span className="text-sm">Form Feedback</span>
              </div>
              <div className="flex items-center space-x-2 text-muted-foreground">
                <BarChart3 className="h-5 w-5" />
                <span className="text-sm">Progress Tracking</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto space-y-8">
          {/* Video Upload Section */}
          <Card className="p-8 bg-fitness-surface/50 backdrop-blur-sm border-fitness-accent/20">
            <CardHeader>
              <CardTitle className="text-3xl font-bold">Upload Your Workout Video</CardTitle>
              <CardDescription className="text-lg">
                Get instant feedback on your exercise form
              </CardDescription>
            </CardHeader>
            <CardContent>
              <VideoUpload 
                onVideoSelect={handleVideoSelect}
                selectedVideo={selectedVideo}
                onClearVideo={handleClearVideo}
              />
            </CardContent>
          </Card>

          {/* Analysis Controls */}
          {selectedVideo && (
            <Card className="p-8 bg-fitness-surface/50 backdrop-blur-sm border-fitness-accent/20">
              <CardHeader>
                <CardTitle className="text-3xl font-bold">Analysis Controls</CardTitle>
                <CardDescription className="text-lg">
                  Process your video to get form feedback
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-col sm:flex-row gap-6">
                  <Button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="flex-1 h-14 text-lg"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="mr-2 h-6 w-6 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Activity className="mr-2 h-6 w-6" />
                        Analyze Form
                      </>
                    )}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={handleClearVideo}
                    className="flex-1 h-14 text-lg"
                  >
                    <RefreshCw className="mr-2 h-6 w-6" />
                    Reset
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Results Section */}
          {analysisResult && (
            <Card className="p-8 bg-fitness-surface/50 backdrop-blur-sm border-fitness-accent/20">
              <CardHeader>
                <CardTitle className="text-3xl font-bold">Analysis Results</CardTitle>
                <CardDescription className="text-lg">
                  Your form analysis and feedback
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-8">
                  {/* Exercise Name */}
                  <div className="flex items-center space-x-4">
                    <div className="w-12 h-12 gradient-fitness rounded-xl flex items-center justify-center">
                      <Target className="h-7 w-7 text-white" />
                    </div>
                    <div>
                      <h3 className="text-2xl font-semibold">Detected Exercise</h3>
                      <p className="text-xl text-muted-foreground">
                        {analysisResult.classified_exercise.split(' ').map(word => 
                          word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
                        ).join(' ')}
                      </p>
                    </div>
                  </div>

                  {/* Detailed Analysis */}
                  {analysisResult.angle_analysis && (
                    <div className="space-y-6">
                      <h3 className="text-2xl font-semibold">Detailed Analysis</h3>
                      <div className="p-6 rounded-xl bg-fitness-surface/30 border border-fitness-accent/10">
                        <pre className="whitespace-pre-wrap text-sm">{analysisResult.angle_analysis.summary}</pre>
                      </div>
                    </div>
                  )}

                  {/* Exercise Tips */}
                  {exerciseTips && (
                    <div className="space-y-6">
                      <h3 className="text-2xl font-semibold">Exercise Tips</h3>
                      <p className="text-lg">{exerciseTips}</p>
                    </div>
                  )}

                  {/* Pose Visualization */}
                  <div className="space-y-6">
                    <h3 className="text-2xl font-semibold">Pose Visualization</h3>
                    <PoseVisualization
                      videoFile={selectedVideo}
                      isAnalyzing={isAnalyzing}
                      processedVideoUrl={processedVideoUrl || undefined}
                    />
                  </div>

                  {/* Joint Analysis */}
                  <div className="space-y-6">
                    <h3 className="text-2xl font-semibold">Joint Analysis</h3>
                    <div className="p-6 rounded-xl bg-fitness-surface/30 border border-fitness-accent/10 max-w-2xl mx-auto">
                      <div className="space-y-4">
                        {Object.entries(analysisResult.angle_analysis.joint_stats).map(([joint, stats]: [string, any]) => (
                          <div key={joint} className="flex items-start space-x-4">
                            <div className="flex-1">
                              <p className="text-lg font-medium mb-2">{joint}</p>
                              <div className="flex items-center justify-between">
                                <div className="grid grid-cols-3 gap-4 text-sm flex-1">
                                  <div className="bg-fitness-surface/50 p-2 rounded-lg">
                                    <p className="text-muted-foreground">Min Angle</p>
                                    <p className="font-medium">{stats.min_angle.toFixed(1)}°</p>
                                  </div>
                                  <div className="bg-fitness-surface/50 p-2 rounded-lg">
                                    <p className="text-muted-foreground">Max Angle</p>
                                    <p className="font-medium">{stats.max_angle.toFixed(1)}°</p>
                                  </div>
                                  <div className="bg-fitness-surface/50 p-2 rounded-lg">
                                    <p className="text-muted-foreground">Range</p>
                                    <p className="font-medium">{stats.range_of_motion.toFixed(1)}°</p>
                                  </div>
                                </div>
                                <span className={`text-sm px-2.5 py-1 rounded-full ml-4 self-center ${
                                  stats.range_of_motion > 0 
                                    ? 'bg-fitness-success/10 text-fitness-success' 
                                    : 'bg-fitness-warning/10 text-fitness-warning'
                                }`}>
                                  {stats.range_of_motion > 0 ? 'Good Range' : 'Limited Range'}
                                </span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default Index;
