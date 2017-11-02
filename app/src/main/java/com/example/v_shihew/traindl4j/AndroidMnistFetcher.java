package com.example.v_shihew.traindl4j;

import android.os.Environment;
import android.util.Log;

import org.apache.commons.codec.android.digest.DigestUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.util.ArchiveUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;


/**
 * Created by v-shihew on 11/1/2017.
 */

public class AndroidMnistFetcher {

    protected File BASE_DIR = new File(MainActivity.ROOT);
    protected static final String LOCAL_DIR_NAME = "MNIST";
    protected File FILE_DIR = new File(BASE_DIR, LOCAL_DIR_NAME);

    private File fileDir;
    private static final String trainingFilesURL = "http://benchmark.deeplearn.online/mnist/train-images-idx3-ubyte.gz";
    private static final String trainingFilesMD5 = "f68b3c2dcbeaaa9fbdd348bbdeb94873";
    private static final String trainingFilesFilename = "train-images-idx3-ubyte.gz";
    public static final String trainingFilesFilename_unzipped = "train-images-idx3-ubyte";
    private static final String trainingFileLabelsURL =
            "http://benchmark.deeplearn.online/mnist/train-labels-idx1-ubyte.gz";
    private static final String trainingFileLabelsMD5 = "d53e105ee54ea40749a09fcbcd1e9432";
    private static final String trainingFileLabelsFilename = "train-labels-idx1-ubyte.gz";
    public static final String trainingFileLabelsFilename_unzipped = "train-labels-idx1-ubyte";

    //Test data:
    private static final String testFilesURL = "http://benchmark.deeplearn.online/mnist/t10k-images-idx3-ubyte.gz";
    private static final String testFilesMD5 = "9fb629c4189551a2d022fa330f9573f3";
    private static final String testFilesFilename = "t10k-images-idx3-ubyte.gz";
    public static final String testFilesFilename_unzipped = "t10k-images-idx3-ubyte";
    private static final String testFileLabelsURL = "http://benchmark.deeplearn.online/mnist/t10k-labels-idx1-ubyte.gz";
    private static final String testFileLabelsMD5 = "ec29112dd5afa0611ce80d1b7f02629c";
    private static final String testFileLabelsFilename = "t10k-labels-idx1-ubyte.gz";
    public static final String testFileLabelsFilename_unzipped = "t10k-labels-idx1-ubyte";


    public File downloadAndUntar() throws IOException {
        if (fileDir != null) {
            return fileDir;
        }

        File baseDir = FILE_DIR;

        Log.i("ddd",FILE_DIR.toString());

        baseDir.mkdirs();

        if (!(baseDir.isDirectory() || baseDir.mkdir())) {
            throw new IOException("Could not mkdir " + baseDir);
        }

        Log.i("ddd","Downloading mnist...");
        // getFromOrigin training records
        File tarFile = new File(baseDir, trainingFilesFilename);
        File testFileLabels = new File(baseDir, testFilesFilename);

        tryDownloadingAFewTimes(new URL(trainingFilesURL), tarFile, trainingFilesMD5);
        tryDownloadingAFewTimes(new URL(testFilesURL), testFileLabels, testFilesMD5);

        ArchiveUtils.unzipFileTo(tarFile.getAbsolutePath(), baseDir.getAbsolutePath());
        ArchiveUtils.unzipFileTo(testFileLabels.getAbsolutePath(), baseDir.getAbsolutePath());

        // getFromOrigin training records
        File labels = new File(baseDir, trainingFileLabelsFilename);
        File labelsTest = new File(baseDir, testFileLabelsFilename);

        tryDownloadingAFewTimes(new URL(trainingFileLabelsURL), labels, trainingFileLabelsMD5);
        tryDownloadingAFewTimes(new URL(testFileLabelsURL), labelsTest, testFileLabelsMD5);

        ArchiveUtils.unzipFileTo(labels.getAbsolutePath(), baseDir.getAbsolutePath());
        ArchiveUtils.unzipFileTo(labelsTest.getAbsolutePath(), baseDir.getAbsolutePath());

        fileDir = baseDir;
        return fileDir;
    }

    private void tryDownloadingAFewTimes(URL url, File f, String targetMD5) throws IOException {
        tryDownloadingAFewTimes(0, url, f, targetMD5);
    }

    private void tryDownloadingAFewTimes(int attempt, URL url, File f, String targetMD5) throws IOException {
        int maxTries = 3;
        boolean isCorrectFile = f.isFile();
        if (attempt < maxTries && !isCorrectFile) {
            FileUtils.copyURLToFile(url, f);
            if (!checkMD5OfFile(targetMD5, f))
                tryDownloadingAFewTimes(attempt + 1, url, f, targetMD5);
        } else if (isCorrectFile) {
            // do nothing, file downloaded
        } else {
            throw new IOException("Could not download " + url.getPath() + "\n properly despite trying " + maxTries
                    + " times, check your connection. File info:" + "\nTarget MD5: " + targetMD5
                    + "\nHash matches: " + checkMD5OfFile(targetMD5, f) + "\nIs valid file: " + f.isFile());
        }
    }

    private boolean checkMD5OfFile(String targetMD5, File file) throws IOException {
        InputStream in = FileUtils.openInputStream(file);
        String trueMd5 = DigestUtils.md5Hex(in);
        IOUtils.closeQuietly(in);
        return (targetMD5.equals(trueMd5));
    }

    public static void gunzipFile(File baseDir, File gzFile) throws IOException {
        Log.i("ddd","gunzip'ing File: " + gzFile.toString());
        Process p = Runtime.getRuntime().exec(String.format("gunzip %s", gzFile.getAbsolutePath()));
        BufferedReader stdError = new BufferedReader(new InputStreamReader(p.getErrorStream()));
        Log.i("ddd","Here is the standard error of the command (if any):\n");
        String s;
        while ((s = stdError.readLine()) != null) {
            Log.i("ddd",s);
        }
        stdError.close();
    }


}

