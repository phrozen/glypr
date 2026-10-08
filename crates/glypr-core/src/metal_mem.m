#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

uint64_t get_metal_current_allocated_bytes() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return (uint64_t)[device currentAllocatedSize];
    }
}
