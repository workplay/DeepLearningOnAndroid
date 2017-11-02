package com.example.v_shihew.traindl4j;

import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

import java.io.IOException;

/**
 * Created by v-shihew on 11/1/2017.
 */

/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

/**
 * Mnist data applyTransformToDestination iterator.
 * @author Adam Gibson
 */
public class AndroidMnistDataSetIterator extends BaseDatasetIterator {

    public AndroidMnistDataSetIterator(int batch, int numExamples) throws IOException {
        this(batch, numExamples, false);
    }

    /**Get the specified number of examples for the MNIST training data set.
     * @param batch the batch size of the examples
     * @param numExamples the overall number of examples
     * @param binarize whether to binarize mnist or not
     * @throws IOException
     */
    public AndroidMnistDataSetIterator(int batch, int numExamples, boolean binarize) throws IOException {
        this(batch, numExamples, binarize, true, false, 0);
    }

    /** Constructor to get the full MNIST data set (either test or train sets) without binarization (i.e., just normalization
     * into range of 0 to 1), with shuffling based on a random seed.
     * @param batchSize
     * @param train
     * @throws IOException
     */
    public AndroidMnistDataSetIterator(int batchSize, boolean train, int seed) throws IOException {
        this(batchSize, (train ? AndroidMnistDataFetcher.NUM_EXAMPLES : AndroidMnistDataFetcher.NUM_EXAMPLES_TEST), false, train,
                true, seed);
    }

    /**Get the specified number of MNIST examples (test or train set), with optional shuffling and binarization.
     * @param batch Size of each patch
     * @param numExamples total number of examples to load
     * @param binarize whether to binarize the data or not (if false: normalize in range 0 to 1)
     * @param train Train vs. test set
     * @param shuffle whether to shuffle the examples
     * @param rngSeed random number generator seed to use when shuffling examples
     */
    public AndroidMnistDataSetIterator(int batch, int numExamples, boolean binarize, boolean train, boolean shuffle,
                                long rngSeed) throws IOException {
        super(batch, numExamples, new AndroidMnistDataFetcher(binarize, train, shuffle, rngSeed));
    }


}
