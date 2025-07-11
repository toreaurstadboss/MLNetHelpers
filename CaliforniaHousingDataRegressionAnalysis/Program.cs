using Microsoft.ML;

namespace CaliforniaHousingDataRegressionAnalysis
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            // California Housing Prices from 1990 census - Kaggle dataset : https://www.kaggle.com/datasets/camnugent/california-housing-prices?resource=download
            var data = context.Data.LoadFromTextFile<HousingData>("./housing.csv", hasHeader: true, separatorChar: ',');

            var split = context.Data.TrainTestSplit(data, testFraction: 0.2); //80% for training - 20% for testing 

            var features = split.TrainSet.Schema
                .Select(col => col.Name)
                .Where(colName => colName != "Label" && colName != "OceanProximity") //only numerical featuers
                .ToArray();

            var pipeline = context.Transforms.Text.FeaturizeText("Text", "OceanProximity")
                .Append(context.Transforms.Concatenate("Features", features))
                .Append(context.Transforms.Concatenate("Feature", "Features", "Text"))
                .Append(context.Regression.Trainers.FastForest());

            var model = pipeline.Fit(split.TrainSet);

            var predictions = model.Transform(split.TestSet);

            var metrics = context.Regression.Evaluate(predictions);

            Console.WriteLine($"R^2 - {metrics.RSquared}");

        }
    }
}
