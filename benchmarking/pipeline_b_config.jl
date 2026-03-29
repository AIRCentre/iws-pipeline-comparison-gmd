module DefaultConfig
    const TARGET_SIZE = (256, 256)
    const RNG_SEED    = 42
    const BATCH_SIZE  = 128

    module DeviceConfig
        const DEVICE = :gpu
    end
end
