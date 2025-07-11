using Microsoft.ML.Data;

namespace SaveAndLoadTrainedModels
{

    public class HousingData
    {

        public float Size { get; set; }

        [VectorType(3)]
        public float[] HistoricalPrices { get; set; } = Array.Empty<float>();

        [ColumnName("Label")]
        public float CurrentPrice { get; set; }

    }

}