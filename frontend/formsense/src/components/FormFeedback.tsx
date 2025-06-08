
import React from 'react';
import { Card } from '@/components/ui/card';
import { Brain, CheckCircle, AlertTriangle, Info } from 'lucide-react';

interface FormFeedbackProps {
  isVisible: boolean;
}

const FormFeedback: React.FC<FormFeedbackProps> = ({ isVisible }) => {
  // Mock feedback data - in real app this would come from your analysis API
  const mockFeedback = {
    summary: "Overall, your squat form shows good fundamentals with room for improvement in knee tracking and depth consistency.",
    detailedFeedback: [
      {
        type: 'positive',
        text: "Excellent back posture maintained throughout the movement. Your spine remains neutral and chest stays up."
      },
      {
        type: 'warning',
        text: "Knee tracking could be improved. Your left knee tends to cave inward slightly during the descent phase."
      },
      {
        type: 'suggestion',
        text: "Try to achieve more consistent depth. Aim for hip crease just below knee level on each repetition."
      },
      {
        type: 'positive',
        text: "Great control on the eccentric (lowering) phase. This helps build strength and prevents injury."
      },
      {
        type: 'warning',
        text: "Watch your weight distribution - you're favoring your right side slightly. Focus on even pressure through both feet."
      }
    ]
  };

  const getIcon = (type: string) => {
    switch (type) {
      case 'positive':
        return <CheckCircle className="h-5 w-5 text-fitness-success" />;
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-fitness-warning" />;
      case 'suggestion':
        return <Info className="h-5 w-5 text-primary" />;
      default:
        return <Brain className="h-5 w-5 text-muted-foreground" />;
    }
  };

  if (!isVisible) {
    return null;
  }

  return (
    <Card className="p-6 animate-fade-in">
      <div className="space-y-6">
        <div className="flex items-center space-x-3">
          <Brain className="h-6 w-6 text-primary" />
          <h3 className="text-lg font-semibold">AI Form Analysis</h3>
        </div>

        {/* Summary */}
        <div className="p-4 bg-muted/50 rounded-lg">
          <h4 className="font-semibold mb-2">Summary</h4>
          <p className="text-sm leading-relaxed">{mockFeedback.summary}</p>
        </div>

        {/* Detailed Feedback */}
        <div className="space-y-4">
          <h4 className="font-semibold">Detailed Analysis</h4>
          <div className="space-y-3">
            {mockFeedback.detailedFeedback.map((feedback, index) => (
              <div key={index} className="flex items-start space-x-3 p-3 bg-fitness-surface rounded-lg border-l-4 border-l-primary/20">
                {getIcon(feedback.type)}
                <p className="text-sm leading-relaxed flex-1">{feedback.text}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Action Items */}
        <div className="p-4 bg-primary/5 rounded-lg border border-primary/20">
          <h4 className="font-semibold mb-2 text-primary">Key Focus Areas</h4>
          <ul className="text-sm space-y-1">
            <li>• Work on knee tracking alignment</li>
            <li>• Practice consistent squat depth</li>
            <li>• Focus on even weight distribution</li>
          </ul>
        </div>
      </div>
    </Card>
  );
};

export default FormFeedback;
