using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
namespace RegressionSolver
{
    public class Program
    {
        private readonly static Settings settings = new();
        private static LinearRegression? model;
        public static async Task Main()
        {
            PrintHeader();
            GetUserInputs();
            PrintSummary();

            var readDataPoints = GetDataPointsInput();
            if (string.IsNullOrEmpty(readDataPoints))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("No data points provided. Exiting...");
                Console.ResetColor();
                return;
            }
            var dataPoints = ParseDataPoints(readDataPoints, settings.Degree);
            model = new LinearRegression(settings);
            await model.TrainAsync(dataPoints, UpdateProgress, settings.Patience);

            await InteractivePredictionLoop(model);
        }

        private static void PrintHeader()
        {
            Console.ForegroundColor = ConsoleColor.DarkGreen;
            Console.WriteLine("Linear Regression Model Trainer");
            Console.WriteLine("Author: Blazej Strus (164489)");
            Console.WriteLine("Date: 2024-03-21");
            Console.WriteLine("Version: 1.0");
            Console.WriteLine("This program trains a linear regression model based on user-provided data\nand allows to predict values, based on trained model.");
            Console.WriteLine();
            Console.ResetColor(); // Reset to default color
        }

        private static void PrintSummary()
        {
            Console.ForegroundColor = ConsoleColor.DarkBlue;
            Console.WriteLine($"Summary: Degree = {settings.Degree}, Epochs = {settings.Epochs}");
            Console.WriteLine();
            Console.ResetColor(); // Reset to default color
        }
        private static void UpdateProgress(int epoch, double loss, string weights, double learningRate)
        {
            Console.SetCursorPosition(0, Console.CursorTop);
            Console.ForegroundColor = ConsoleColor.DarkYellow;
            string learningRateFormatted = $"{learningRate:E2}".ToLower().Replace("1,00", "1").Replace("1.00", "1").Replace("e+00", "e+0").Replace("e-00", "e-0").Replace("e+0", "e+").Replace("e-0", "e-");
            string formatedLoss;
            if (((int)loss).ToString().Length > 5)
            {
                formatedLoss = $"..." + ((int)loss).ToString()[^5..];
            }
            else
            {
                formatedLoss = $"{loss,10:F5}";
            }

            string progress = $"Epoch {epoch,7}: Loss = {formatedLoss} | Weights: {weights} | Learning Rate: {learningRateFormatted}";
            Console.Write(progress.PadRight(Console.WindowWidth - 1));
            Console.ResetColor();
        }

        private static void GetUserInputs()
        {
            settings.Degree = InputInt($"Enter the degree of polynomial features (1 - for linear, 2 for quadratic, etc.), (default = {settings.Degree}):", settings.Degree);
            settings.Epochs = InputInt($"Enter the number of epochs (default = {settings.Epochs}):", settings.Epochs);
            settings.Patience = InputInt($"Enter the number of epochs to wait before reducing the learning rate (default patience = {settings.Patience}):", settings.Patience);
        }

        private static int InputInt(string prompt, int defaultValue)
        {
            Console.WriteLine(prompt);
            return int.TryParse(Console.ReadLine(), out int value) ? value : defaultValue;
        }

        private static string? GetDataPointsInput()
        {
            Console.WriteLine("Enter data points (x1,x2,...,xn,y; x1,x2,...,xn,y; ...):");
            return Console.ReadLine();
        }

        private static async Task InteractivePredictionLoop(LinearRegression model)
        {
            while (true)
            {
                Console.WriteLine("Enter new x values to predict y (leave empty and press enter to exit):");
                var newXInput = Console.ReadLine();
                if (string.IsNullOrEmpty(newXInput)) { break; }

                var newXValues = newXInput.Split(',').Select(n => double.TryParse(n, out double x) ? x : 0).ToArray();
                var augmentedXValues = AugmentFeatures(newXValues, settings.Degree);
                var predictedY = model.Predict(augmentedXValues);
                Console.WriteLine($"Predicted y: {predictedY}");
            }
            await Task.CompletedTask;
        }

        private static List<DataPoint> ParseDataPoints(string input, int degree)
            => input.Split(';')
                    .Select(point => point.Split(',')
                                          .Select(s => double.Parse(s, CultureInfo.InvariantCulture))
                                          .ToArray())
                    .Select(values => new DataPoint(AugmentFeatures(values[..^1], degree), values[^1]))
                    .ToList();

        private static double[] AugmentFeatures(double[] features, int degree)
            => features.SelectMany(f => Enumerable.Range(1, degree).Select(pow => Math.Pow(f, pow))).ToArray();
    }
}
