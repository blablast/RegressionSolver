using System;
using System.Collections.Generic;
using System.Linq;
using System.Globalization;
using System.Text;
using System.Threading.Tasks;

namespace RegressionSolver
{
    internal class LinearRegression
    {
        public double[] Weights { get; private set; }
        public List<double> LossHistory = new();
        public double LearningRate { get; set; }
        public double MinLearningRate { get; set; }
        public int Epochs { get; set; }

        private CultureInfo culture = CultureInfo.InvariantCulture; // Using InvariantCulture

        public LinearRegression(Settings settings)
        {

            Epochs = settings.Epochs > 0 ? settings.Epochs : throw new ArgumentOutOfRangeException(nameof(settings.Epochs));
            LearningRate = settings.LearningRate;
            MinLearningRate = settings.MinLearningRate;
            Weights = Array.Empty<double>();
        }

        // Computes the mean squared error loss
        private double ComputeLoss(IEnumerable<DataPoint> dataPoints)
            => dataPoints.Average(dataPoint => Math.Pow(dataPoint.Label - Predict(dataPoint.Features), 2));

        // Makes a prediction based on the input features and learned weights
        public double Predict(double[] features)
            => Math.Round(Weights[0] + features.Select((t, i) => t * Weights[i + 1]).Sum(), 10);

        // Trains the model using the provided data points
        public async Task TrainAsync(List<DataPoint> dataPoints, Action<int, double, string, double> updateProgress, int patience)
        {
            int featureCount = dataPoints.First().Features.Length;
            var random = new Random();
            Weights = Enumerable.Range(0, featureCount + 1).Select(_ => random.NextDouble() * 0.01 - 0.005).ToArray();


            string numberFormat = "F5";
            double previousLoss = double.MaxValue;
            int patienceCounter = 0;
            var minLearningRate = 1e-7;

            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                UpdateWeights(dataPoints);
                double totalLoss = ComputeLoss(dataPoints);
                LossHistory.Add(totalLoss);

                if (totalLoss >= previousLoss)
                {
                    patienceCounter++;
                    if (patienceCounter >= patience)
                    {
                        if (LearningRate == minLearningRate)
                        {
                            updateProgress(epoch, totalLoss, string.Join(", ", Weights.Select(w => w.ToString(numberFormat, culture))), LearningRate);
                            PrintError("Training stopped due to patience limit.");
                            break;
                        }
                        else
                        {
                            LearningRate = Math.Max(LearningRate / 10, MinLearningRate);
                            patienceCounter = 0;
                        }
                    }
                }
                else
                {
                    patienceCounter = 0;
                }

                if (totalLoss < previousLoss)
                {
                    updateProgress(epoch, totalLoss, string.Join(", ", Weights.Select(w => w.ToString(numberFormat, culture))), LearningRate);
                }

                previousLoss = totalLoss;

                if (double.IsNaN(totalLoss))
                {
                    //PrintError("Training stopped due to NaN loss.");
                    //break;
                }

            }

            Console.WriteLine();  // Ensure the final output starts on a new line
            DisplayFinalWeights(numberFormat);
            await Task.CompletedTask;
        }

        private static void PrintError(string message)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"\n{message}");
            Console.ResetColor();
        }
        // Updates the weights based on gradients calculated from the data points
        private void UpdateWeights(List<DataPoint> dataPoints)
        {
            double[] totalGradients = new double[Weights.Length];

            foreach (var dataPoint in dataPoints)
            {
                double error = dataPoint.Label - Predict(dataPoint.Features);
                totalGradients[0] += -2 * error;  // Gradient for bias
                for (int i = 1; i < Weights.Length; i++)
                {
                    totalGradients[i] += -2 * error * dataPoint.Features[i - 1];  // Gradient for each feature weight
                }
            }

            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] -= (LearningRate / dataPoints.Count) * totalGradients[i];  // Update weights
            }
        }

        // Displays the final weights after training
        private void DisplayFinalWeights(string numberFormat)
        {
            // Constructing the equation string
            StringBuilder equation = new("y = ");
            for (int i = 0; i < Weights.Length; i++)
            {
                // Append each weight and its corresponding x term, excluding the first weight (intercept)
                equation.Append((i > 0)
                    ? $" {(Weights[i] >= 0 ? "+" : "-")} {Math.Abs(Weights[i]).ToString(numberFormat, culture)} * x^{i}"
                    : $"{Weights[i].ToString(numberFormat, culture)}");
            }

            StringBuilder sbHeader = new StringBuilder("|");
            StringBuilder sbValues = new StringBuilder("| ");
            var j = 0;
            foreach (var weight in Weights)
            {
                sbHeader.Append($" w{j.ToString().PadRight(12)}|");
                sbValues.Append(weight.ToString(numberFormat, culture).PadLeft(12)).Append(" | ");
                j++;
            }

            Console.ForegroundColor = ConsoleColor.DarkGreen;
            Console.WriteLine($"Equation: {equation}");
            Console.WriteLine("Final weights:");
            Console.WriteLine(sbHeader.ToString());
            Console.WriteLine(sbValues.ToString());
            Console.ResetColor();

        }
    }
}
