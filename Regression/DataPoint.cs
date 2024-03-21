using System;

namespace RegressionSolver
{
    public class DataPoint
    {
        public double[] Features { get; }
        public double Label { get; }

        public DataPoint(double[] features, double label)
        {
            Features = features ?? throw new ArgumentNullException(nameof(features));
            Label = label;
        }
    }
}