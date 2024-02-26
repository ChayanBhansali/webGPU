async function computeSum(vectorSize,batchSize,totalThreads,device) {
    
    const vectorA = new Float32Array(vectorSize);
    const vectorB = new Float32Array(vectorSize);
    const sumVector = new Float32Array(vectorSize);
    
    // Initialization of Vectors
    for (let i = 0; i < vectorSize; i++) {
        vectorA[i] = Math.random();
        vectorB[i] = Math.random();
        
    }

    // time it took for CPU addition :
    const startTimeCPU = performance.now();

    // Addition of Vector on CPU
    for (let i = 0; i < vectorSize; i++) {
        sumVector[i] = vectorA[i] + vectorB[i]; // Calculate the sum and store it in sumVector
    }
    console.log("Vector A:", vectorA);
    console.log("Vector B:", vectorB);
    console.log("Sum Vector:", sumVector);

    const executionTimeCPU = performance.now() - startTimeCPU;

    // Creation of GPU buffers :
    // Create GPU buffers for vectors A, B, and C
    const bufferA = device.createBuffer({
        size: vectorSize * Float32Array.BYTES_PER_ELEMENT, // allocate size of 10,000 elements
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC , //usage specifies how the buffer will be used. STORAGE indicates that the buffer will be used for storage operations (read/write), and COPY_SRC indicates that the buffer can be used as the source of a copy operation.
        mappedAtCreation: true,
    }); // COPY_SRC means it will be used as the source of copy operation, matlab yaah se copy hoga
    new Float32Array(bufferA.getMappedRange()).set(vectorA); // copy vectorA to gpu
    bufferA.unmap();

    const bufferB = device.createBuffer({
        size: vectorSize * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Float32Array(bufferB.getMappedRange()).set(vectorB);
    bufferB.unmap();

    const bufferC = device.createBuffer({
        size: vectorSize * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    //Creating Binding Group Layout :
    // Create a binding group layout
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0, // Binding index for bufferA
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage",
                },
            },
            {
                binding: 1, // Binding index for bufferB
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage",
                },
            },
            {
                binding: 2, // Binding index for bufferC
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                },
            },
        ],
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
        {
            binding: 0,
            resource: {
            buffer: bufferA
            }
        },
        {
            binding: 1,
            resource: {
            buffer: bufferB
            }
        },
        {
            binding: 2,
            resource: {
            buffer: bufferC
            }
        }
        ]
    });


    // Creating Shader Module
    // and writing the WSGL shader code :

    const shaderModule = device.createShaderModule({
        code: `
            struct Vector {
                values: array<f32>,
            }
            // var<private> variable_name : dataType;
            // source : https://www.w3.org/TR/WGSL/#vector-types
            // this arrayLength debug : https://google.github.io/tour-of-wgsl/types/arrays/runtime-sized-arrays/
            @group(0) @binding(0) var<storage, read> firstVector : Vector;
            @group(0) @binding(1) var<storage, read> secondVector : Vector;
            @group(0) @binding(2) var<storage, read_write> resultVector : Vector;
    
            @compute @workgroup_size(${batchSize})
            fn main(@builtin(global_invocation_id) global_id : vec3u) {
                let index = global_id.x ;
                if (index >= arrayLength(&firstVector.values)) {
                    return;
                }
    
                resultVector.values[index] = firstVector.values[index] + secondVector.values[index];
            }
        `
    });


    // creating Compute Pipeline to run WSGL code on GPU: 
    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
        module: shaderModule,
        entryPoint: "main"
        }
    });


    // Setting up the code via the pipeline on gpu :
    const commandEncoder = device.createCommandEncoder();

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    const workgroupCountX = totalThreads;
    // const workgroupCountY = 1; // Since this is a 1D calculation
    passEncoder.dispatchWorkgroups(workgroupCountX);
    passEncoder.end();


    // As the bufferC which stores the result can't be
    // printed on console or READ
    // we create a 4th buffer to do so :


    // Get a GPU buffer for reading in an unmapped state.
    const gpuReadBuffer = device.createBuffer({
        size: vectorSize * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // now copy the BufferC to the 4th Buffer : 
    // Encode commands for copying buffer to buffer.
    commandEncoder.copyBufferToBuffer(
        bufferC /* source buffer */,
        0 /* source offset */,
        gpuReadBuffer /* destination buffer */,
        0 /* destination offset */,
        vectorSize * Float32Array.BYTES_PER_ELEMENT /* size */
    );

    // Run the sequence of instructions :
    // Submit GPU commands.
    const gpuCommands = commandEncoder.finish();

    // start the timer :
    const startTime = performance.now();

    device.queue.submit([gpuCommands]);

    // print the result :
    // Read buffer.
    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = gpuReadBuffer.getMappedRange();

    // end the timer :
    const executionTime = performance.now() - startTime;
    
    console.log(new Float32Array(arrayBuffer));
    return {executionTime,executionTimeCPU};

}


async function requestGPU() {
    if ('gpu' in navigator) {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) { 
            document.getElementById('gpuStatus').textContent = 'GPU Adapter no available.';
            return; 
        }
        const device = await adapter.requestDevice();
        document.getElementById('gpuStatus').textContent = "We're able to access GPU";

        // For Different Batch Sizes :
        const vectorSize = 10000;
        const batchSizes = [2, 3, 4, 5, 6, 7, 8, 9, 10];
        var executionTimes = [];

        for (let i = 0; i < batchSizes.length; i++) {
            const batchSize = batchSizes[i];
            const threads = Math.ceil(vectorSize / batchSize);
            const timeTaken = await computeSum(vectorSize, batchSize, threads, device);
            executionTimes.push(timeTaken.executionTime);
        }

        console.log("Batch Sizes:", batchSizes);
        console.log("Execution Times:", executionTimes);
        createBatchSizeTimeChart([batchSizes,executionTimes]);


        // For Different Vector Sizes :
        const vectorSizes = [10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000];
        const batchSize = 3;
        executionTimes = [];
        for (let i = 0; i < vectorSizes.length; i++) {
            const vectorSize = vectorSizes[i]
            const threads = Math.ceil(vectorSize / batchSize);
            const timeTaken = await computeSum(vectorSize, batchSize, threads, device);
            executionTimes.push(timeTaken.executionTime);
        }

        console.log("Vector Sizes:", vectorSizes);
        console.log("Execution Times:", executionTimes);
        createVectorSizeTimeChart([vectorSizes,executionTimes]);

    }
}

// Call the async function
requestGPU();

function createBatchSizeTimeChart(data) {
    var ctx = document.getElementById('batchSizeTime').getContext('2d');
    var batchSizeTime = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data[0],
            datasets: [{
                label: 'Execution Time vs Batch Size',
                data: data[1],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: false,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Batch Size'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Execution Time (ms)'
                    }
                }
            }
        }
    });
}


function createVectorSizeTimeChart(data) {
    var ctx = document.getElementById('vectorSizeTime').getContext('2d');
    var vectorSizeTime = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data[0],
            datasets: [{
                label: 'Execution Time vs Vector Size',
                data: data[1],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: false,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Vector Size'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Execution Time (ms)'
                    }
                }
            }
        }
    });
}

