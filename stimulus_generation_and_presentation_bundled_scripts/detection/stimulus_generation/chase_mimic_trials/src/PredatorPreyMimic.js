import { 
    PsyanimConstants,

    PsyanimPredatorPrefab,
    PsyanimPredatorAgent,

    PsyanimPreyPrefab,
    PsyanimPreyAgent,

    PsyanimMimic

} from 'psyanim2';

export default {
    key: 'PredatorPreyMimic',
    wrapScreenBoundary: false,
    entities: [
        {
            name: 'predator',
            initialPosition: { x: 100, y: 100 },
            shapeParams: {
                shapeType: PsyanimConstants.SHAPE_TYPE.CIRCLE,
                radius: 12, color: 0xff0000
            },
            prefab: { 
                type: PsyanimPredatorPrefab,
                params: {
                    maxChaseSpeed: 1.5, //0.75, //3.0, //1.5,
                    maxChaseAcceleration: 0.1,
                    maxWanderSpeed: 1.5, //3.0, //1.5,
                    maxWanderAcceleration: 0.1,
                    boredomDistance: 500,
                    showDebugLogs: true,
                    subtlety: 30
                }
            },
            components: [
                {
                    type: PsyanimPredatorAgent,
                    params: {
                        target: {
                            entityName: 'prey'
                        }
                    }    
                }
            ]
        },
        {
            name: 'prey',
            initialPosition: { x: 700, y: 500 },
            shapeParams: {
                shapeType: PsyanimConstants.SHAPE_TYPE.CIRCLE,
                radius: 12, color: 0xffff00
                //0x0000ff 
            },
            prefab: { 
                type: PsyanimPreyPrefab,
                params: {
                    maxFleeSpeed: 1.8, //3.6, //1.8,
                    maxFleeAcceleration: 0.15,
                    maxWanderSpeed: 1.5, //3.0, //1.5,
                    maxWanderAcceleration: 0.1,
                    showDebugLogs: true,
                    safetyDistance: 100, //250,

                }
            },
            components: [
                {
                    type: PsyanimPreyAgent,
                    params: {
                        target: {
                            entityName: 'predator'
                        }
                    }    
                }
            ]
        },
        {
            name: 'preyMimic',
            shapeParams: {
                shapeType: PsyanimConstants.SHAPE_TYPE.CIRCLE,
                radius: 12, color: 0xcccccc, depth: 1
            },
            matterOptions: {
                isSensor: true
            },
            components: [
                { 
                    type: PsyanimMimic,
                    params: {
                        target: {
                            entityName: 'prey',
                        },
                        angleOffset: 0
                    }
                }
            ]
        }
    ],
}