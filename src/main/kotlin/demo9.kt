import io.jenetics.DoubleChromosome
import io.jenetics.DoubleGene
import io.jenetics.Genotype
import io.jenetics.engine.Engine
import io.jenetics.engine.EvolutionResult
import io.jenetics.engine.EvolutionStatistics
import io.jenetics.engine.Limits
import io.jenetics.prog.regression.LossFunction
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.cpu.nativecpu.NDArray

fun main() {
    val max = 1.0

    val layers : List<DenseLayer> = listOf(
        DenseLayer.Builder().nIn(2).nOut(3).build(),
        DenseLayer.Builder().nIn(3).nOut(3).build(),
        DenseLayer.Builder().nIn(3).nOut(1).build(),
    )

    val config = NeuralNetConfiguration.Builder()
        .activation(Activation.RELU)
        .list(*layers.toTypedArray())
        .build()

    fun genotypeNeuralFactory(layers : Iterable<DenseLayer>) : Genotype<DoubleGene> {
        val chromosomes = layers.flatMap { layer -> (0 .. (layer.nIn)).toList().map { DoubleChromosome.of(-max, max, layer.nOut.toInt()) } }
        return Genotype.of(chromosomes)
    }

    fun neuralNetworkFromGenotype(layers : Iterable<DenseLayer>, genotype: Genotype<DoubleGene>) : MultiLayerNetwork {
        val slicing = layers.map { it.nIn.toInt() + 1 /* bias */ }.scan(0) { acc, i -> acc + i }
        val chromosomeGroupLayers = slicing.zipWithNext().map { (start, end) -> genotype.toList().subList(start, end) }
        val layersWeights = chromosomeGroupLayers.map { group -> group.map { it.toList().map { gene -> gene.allele().toFloat() } } }
        val linearWeight = layersWeights.map { weight -> weight.flatten() }.flatten()
        val network = MultiLayerNetwork(config)
        network.init(NDArray(linearWeight.toFloatArray()), false)
        return network
    }

    val dataset = listOf(
        Pair((floatArrayOf(1f, 0f)), 1.0),
        Pair((floatArrayOf(1f, 1f)), 1.0),
        Pair((floatArrayOf(0f, 1f)), 1.0),
        Pair((floatArrayOf(0f, 0f)), 0.0),
    )

    fun fitness(genotype: Genotype<DoubleGene>) : Double {
        val neuralNetwork = neuralNetworkFromGenotype(layers, genotype)
        val input = dataset.map { it.first }.toTypedArray()
        val datasetForNetwork = NDArray(input)
        val rightResult = neuralNetwork.output(datasetForNetwork, false).toDoubleVector().toTypedArray()
        return LossFunction.mse(rightResult, dataset.map { it.second }.toTypedArray())
    }

    val engine = Engine.builder(::fitness, genotypeNeuralFactory(layers))
        .minimizing()
        .executor(Runnable::run)
        .build()

    val statistics = EvolutionStatistics.ofNumber<Double>()
    val result = engine
        .stream()
        .limit(Limits.bySteadyFitness(10))
        .peek(statistics)
        .collect(EvolutionResult.toBestPhenotype())

    val net = neuralNetworkFromGenotype(layers, result.genotype())

    println(statistics)
    println(net.feedForward(NDArray(floatArrayOf(1f, 0f)), false))
    println(net.feedForward(NDArray(floatArrayOf(0f, 0f)), false))
}