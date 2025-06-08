
import React from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

const exercises = [
  { id: 'squat', name: 'Squat', icon: '🏋️' },
  { id: 'pushup', name: 'Push-up', icon: '💪' },
  { id: 'deadlift', name: 'Deadlift', icon: '🏋️‍♀️' },
  { id: 'lunge', name: 'Lunge', icon: '🦵' },
  { id: 'plank', name: 'Plank', icon: '🧘‍♀️' },
  { id: 'pullup', name: 'Pull-up', icon: '🔄' },
];

interface ExerciseSelectorProps {
  selectedExercise: string | null;
  onExerciseSelect: (exerciseId: string) => void;
}

const ExerciseSelector: React.FC<ExerciseSelectorProps> = ({ selectedExercise, onExerciseSelect }) => {
  return (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">Select Exercise Type</h3>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {exercises.map((exercise) => (
          <Button
            key={exercise.id}
            variant={selectedExercise === exercise.id ? "default" : "outline"}
            className="h-auto p-4 flex flex-col items-center space-y-2"
            onClick={() => onExerciseSelect(exercise.id)}
          >
            <span className="text-2xl">{exercise.icon}</span>
            <span className="text-sm font-medium">{exercise.name}</span>
          </Button>
        ))}
      </div>
    </Card>
  );
};

export default ExerciseSelector;
