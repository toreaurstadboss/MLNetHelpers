using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Onnx;

namespace SaveAndLoadTrainedModels
{
    internal class Program
    {
        static void Main(string[] args)
        {
            HousingData[] housingData = new HousingData[]
            {
                new HousingData
                {
                    Size = 600f,
                    HistoricalPrices = new float[] { 100000f, 125000f, 122000f },
                    CurrentPrice = 170000f
                },
                new HousingData
                {
                    Size = 1000f,
                    HistoricalPrices = new float[] { 200000f, 250000f, 230000f },
                    CurrentPrice = 225000f
                },
                new HousingData
                {
                    Size = 1000f,
                    HistoricalPrices = new float[] { 126000f, 130000f, 200000f },
                    CurrentPrice = 195000f
                }

            };

            //Create MLContext 
            MLContext mlContext = new MLContext();

            // Load Data 
            IDataView data = mlContext.Data.LoadFromEnumerable(housingData);

            // Define data preparation estimator 
            EstimatorChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> pipelineEstimator =
                mlContext.Transforms            
                .Concatenate("Features", new string[] { "Size", "HistoricalPrices" })
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.Regression.Trainers.Sdca());

            // train model - regression model
            ITransformer trainedModel = pipelineEstimator.Fit(data);

            // Save model (default format)
            mlContext.Model.Save(trainedModel, data.Schema, "model.zip");

            Console.WriteLine($"Saved ML.Net model to output directory: {Path.Combine(Environment.CurrentDirectory, "model.zip")}");

            // Save ONNX model (ONNX = Open Neural Network Exchange)    

            using (FileStream stream = File.Create("onnx_model.onnx"))
            {
                mlContext.Model.ConvertToOnnx(trainedModel, data, stream);
            }

            Console.WriteLine($"Saved ONNX ML.Net model to output directory: {Path.Combine(Environment.CurrentDirectory, "onnx_model.onnx")}");

            //Define DataViewSchema for data preparation pipeline and trained model 
            DataViewSchema modelSchema;

            // Load trained model
            ITransformer loadedTrainedModel = mlContext.Model.Load("model.zip", out modelSchema);

            // Load in new data
            HousingData[] newHousingData = new HousingData[]
            {
                new()
                {
                    Size = 1000f,
                    HistoricalPrices = new[] { 300_000f, 350_000f, 450_000f },
                    CurrentPrice = 550_00f
                }
            };

            // Load in Onnx model
            OnnxScoringEstimator estimator = mlContext.Transforms.ApplyOnnxModel("onnx_model.onnx");           

            IDataView newHousingDataView = mlContext.Data.LoadFromEnumerable(newHousingData);

            // Fitted model using the saved ONNX model from previous training
            var loadedModel = estimator.Fit(newHousingDataView);

           

        }

    }
}
