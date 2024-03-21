using System;

namespace RegressionSolver
{
    internal class Settings
    {
        private int degree = 1;
        public int Patience { get; set; } = 6;
        public int Degree
        {
            get => degree;
            set
            {
                degree = value;
                LearningRate = Math.Pow(1e-2, degree);
                MinLearningRate = 1e-5 * LearningRate;
                Epochs = (int)(1000 * Math.Pow(10, degree));
            }
        }
        public int Epochs { get; set; } = 10000;
        public double LearningRate { get; set; }
        public double MinLearningRate { get; set; }
    }
}
