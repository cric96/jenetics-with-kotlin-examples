import io.jenetics.*
import io.jenetics.engine.Engine
import io.jenetics.engine.EvolutionResult
import io.jenetics.engine.EvolutionStatistics
import io.jenetics.engine.Limits
import io.jenetics.prog.regression.LossFunction
import io.jenetics.util.RandomRegistry
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.cpu.nativecpu.NDArray
import kotlin.streams.toList

/* similar to example 9, but there I try to learn the min function*/
fun main() {
    val max = 1.0

    val layers : List<DenseLayer> = listOf(
        DenseLayer.Builder().nIn(2).nOut(4).build(),
        DenseLayer.Builder().nIn(4).nOut(2).build(),
        DenseLayer.Builder().nIn(2).nOut(1).build(),
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

    val random = RandomRegistry.random()
    val dataset = (1..100).map { floatArrayOf(random.nextInt(1000).toFloat(), random.nextInt(1000).toFloat()) }
        .map { Pair(it, it.minOrNull()?.toDouble()) }

    fun fitness(genotype: Genotype<DoubleGene>) : Double {
        val neuralNetwork = neuralNetworkFromGenotype(layers, genotype)
        val input = dataset.map { it.first }.toTypedArray()
        val datasetForNetwork = NDArray(input)
        val rightResult = neuralNetwork.output(datasetForNetwork, false).toDoubleVector().toTypedArray()
        return LossFunction.mse(rightResult, dataset.map { it.second }.toTypedArray())
    }

    val engine = Engine.builder(::fitness, genotypeNeuralFactory(layers))
        .minimizing()
        .populationSize(1000)
        .build()

    val statistics = EvolutionStatistics.ofNumber<Double>()
    var generation = 0
    val result = engine
        .stream()
        .limit(Limits.bySteadyFitness(200))
        .peek { generation += 1; println("generation $generation") }
        .peek(statistics)
        .collect(EvolutionResult.toBestPhenotype())

    val net = neuralNetworkFromGenotype(layers, result.genotype())

    println(statistics)
    (1..100).map { floatArrayOf(random.nextInt(1000).toFloat(), random.nextInt(1000).toFloat()) }
        .map { NDArray(it) }
        .forEach { println(net.feedForward(it, false)) }
}