using Microsoft.ML.Data;

namespace SentimentAnalysis
{

    public class SentimentPrediction : SentimentData
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        /// <summary>
        /// NOTE : The probability that the Prediciton = 1 (Positive sentiment).
        /// </summary>
        public float Probability { get; set; }

        public float Score { get; set; }

    }

}
