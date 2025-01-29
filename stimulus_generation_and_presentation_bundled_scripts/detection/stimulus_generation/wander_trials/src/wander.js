import {
    PsyanimConstants,
    PsyanimWanderAgentPrefab,
} from 'psyanim2';

// adapted from https://github.com/ZishanSu/psyanim2-testing/blob/main/src/PredatorPrey.js

// In this scene with 2 agents wandering, I'm going to call one the pseudoPredator and one the pseudoPrey - they're both wandering, 
// but the pseudoPredator will be a little slower than the pseudoPrey in the final videos in the index.js files
// (they're kept same during scene definition here for simplicity)

export default {
    key: 'wanderScene',
    entities: [{
            name: 'pseudoPredator',
            initialPosition: { x: 300, y: 300 },
            shapeParams: {
                shapeType: PsyanimConstants.SHAPE_TYPE.CIRCLE,
                radius: 12,
                color: 0x000000
            },
            prefab: {
                type: PsyanimWanderAgentPrefab,
                params: {
                    maxWanderSpeed: 1.5,
                    maxWanderAcceleration: 0.1,
                    maxAngleChangePerFrame: 35,
                    minScreenBoundaryDistance: 50,
                    debug: false
                }
            },
        },
        {
            name: 'pseudoPrey',
            initialPosition: { x: 500, y: 300 },
            shapeParams: {
                shapeType: PsyanimConstants.SHAPE_TYPE.CIRCLE,
                radius: 12,
                color: 0xcccccc
            },
            prefab: {
                type: PsyanimWanderAgentPrefab,
                params: {
                    maxWanderSpeed: 1.8, //1.8,
                    maxWanderAcceleration: 0.15, //0.15,
                    maxAngleChangePerFrame: 35,
                    minScreenBoundaryDistance: 50,
                    debug: false
                }
            },
        }
    ]
}