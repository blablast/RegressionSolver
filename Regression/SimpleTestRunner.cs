
//namespace RegressionSolver;

//public class SimpleTestRunner
//{
//    public static async Task LinearFunctionTest()
//    {
//        var dataPoints = Enumerable.Range(1, 10)
//                                   .Select(x => new DataPoint(new[] { (double)x }, 2 * x + 1))
//                                   .ToList();
//        var model = new LinearRegression(0.01, 1000, 2);
//        await model.TrainAsync(dataPoints);

//        bool isTestPassed = Math.Abs(model.Weights[1] - 2) < 0.1 && Math.Abs(model.Weights[0] - 1) < 0.1;
//        Console.WriteLine($"LinearFunctionTest Passed: {isTestPassed}");
//    }

//    public static async Task QuadraticFunctionTest()
//    {
//        var dataPoints = Enumerable.Range(1, 10)
//                                   .Select(x => new DataPoint(new[] { (double)x, Math.Pow(x, 2) }, Math.Pow(x, 2)))
//                                   .ToList();
//        var model = new LinearRegression(0.0001, 10000, 4);
//        await model.TrainAsync(dataPoints);

//        bool isTestPassed = Math.Abs(model.Weights[2] - 1) < 0.1 && Math.Abs(model.Weights[1]) < 0.1;
//        Console.WriteLine($"QuadraticFunctionTest Passed: {isTestPassed}");
//    }
//}
