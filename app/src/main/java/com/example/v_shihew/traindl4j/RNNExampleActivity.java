package com.example.v_shihew.traindl4j;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.InputStreamInputSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

public class RNNExampleActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_rnnexample);

        TextView tv2 = (TextView) findViewById(R.id.textview2);
        tv2.setText("dddd1234567dddd");

        try {
            trainRNN();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }

    private void trainRNN() throws IOException, InterruptedException {

        int miniBatchSize = 32;

        // ----- Load the training data -----
        RecordReader trainReader = new CSVRecordReader(0, ";");
//        trainReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath() + "/passengers_train_%d.csv", 0, 0));
        trainReader.initialize(new InputStreamInputSplit(getAssets().open("passengers_train_0.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainReader,miniBatchSize,-1,-1,true)


        RecordReader testReader = new CSVRecordReader(0, ";");
//        testReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath() + "/passengers_test_%d.csv", 0, 0));
        testReader.initialize(new InputStreamInputSplit(getAssets().open("passengers_test_0.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(testReader, miniBatchSize, -1,-1);


        Log.i("ddd",String.valueOf(trainIter.hasNext()));

        //Create data set from iterator here since we only have a single data set
        DataSet trainData = trainIter.next();
        DataSet testData = testIter.next();

        //Normalize data, including labels (fitLabel=true)
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        normalizer.fitLabel(true);
        normalizer.fit(trainData);              //Collect training data statistics

        normalizer.transform(trainData);
        normalizer.transform(testData);


        // ----- Configure the network -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(140)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS)
                .learningRate(0.0015)
                .list()
                .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10)
                        .build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY).nIn(10).nOut(1).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new ScoreIterationListener(20));

        // ----- Train the network, evaluating the test set performance at each epoch -----
        int nEpochs = 300;

        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainData);
            Log.i("ddd","Epoch " + i + " complete. Time series evaluation:");

            //Run regression evaluation on our single column input
            RegressionEvaluation evaluation = new RegressionEvaluation(1);
            INDArray features = testData.getFeatureMatrix();

            INDArray lables = testData.getLabels();
            INDArray predicted = net.output(features, false);

            evaluation.evalTimeSeries(lables, predicted);

            //Just do sout here since the logger will shift the shift the columns of the stats
            Log.i("ddd",evaluation.stats());
        }

        //Init rrnTimeStemp with train data and predict test data
        net.rnnTimeStep(trainData.getFeatureMatrix());
        INDArray predicted = net.rnnTimeStep(testData.getFeatureMatrix());

        //Revert data back to original values for plotting
        normalizer.revert(trainData);
        normalizer.revert(testData);
        normalizer.revertLabels(predicted);

        INDArray trainFeatures = trainData.getFeatures();
        INDArray testFeatures = testData.getFeatures();

        /*
        //Create plot with out data
        XYSeriesCollection c = new XYSeriesCollection();
        createSeries(c, trainFeatures, 0, "Train data");
        createSeries(c, testFeatures, 99, "Actual test data");
        createSeries(c, predicted, 100, "Predicted test data");
        */
        //plotDataset(c);

        Log.i("ddd","----- Example Complete -----");
    }
/*
    private static void createSeries(XYSeriesCollection seriesCollection, INDArray data, int offset, String name) {
        int nRows = data.shape()[2];
        XYSeries series = new XYSeries(name);
        for (int i = 0; i < nRows; i++) {
            series.add(i + offset, data.getDouble(i));
        }
        seriesCollection.addSeries(series);
    }
    */

}










/**
 * Generate an xy plot of the datasets provided.
 */

/*
private static void plotDataset(XYSeriesCollection c) {

        String title = "Regression example";
        String xAxisLabel = "Timestep";
        String yAxisLabel = "Number of passengers";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = true;
        boolean tooltips = false;
        boolean urls = false;
        JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls);

// get a reference to the plot for further customisation...
final XYPlot plot = chart.getXYPlot();

// Auto zoom to fit time series in initial window
final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setAutoRange(true);

        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Training Data");

        RefineryUtilities.centerFrameOnScreen(f);
        f.setVisible(true);
        }
        }
*/