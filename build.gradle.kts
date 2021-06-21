import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

val jeneticsVersion = "6.2.0"
plugins {
    kotlin("jvm") version "1.5.10"
    application
}

group = "me.cric"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("io.jenetics", "jenetics", jeneticsVersion)
    implementation("io.jenetics", "jenetics.prog", jeneticsVersion)
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnit()
}

tasks.withType<KotlinCompile>() {
    kotlinOptions.jvmTarget = "1.8"
}

application {
    mainClassName = "MainKt"
}