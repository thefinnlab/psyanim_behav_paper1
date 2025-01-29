import { 
    PsyanimConstants,
    PsyanimMouseFollowTarget,
    PsyanimArriveAgentPrefab,
    PsyanimArriveAgent
} from 'psyanim2';

export default {
    key: 'MyArriveAgentPrefabScene',
    entities: [
        {
            name: 'mouseFollowTarget',
            initialPosition: { x: 400, y: 300 },
            shapeParams: {
                shapeType: PsyanimConstants.SHAPE_TYPE.CIRCLE,
                radius: 4,
                color: 0x00ff00
            },
            components: [
                { type: PsyanimMouseFollowTarget }
            ]
        },
        {
            name: 'agent1',
            initialPosition: { x: 600, y: 450 },
            shapeParams: {
                shapeType: PsyanimConstants.SHAPE_TYPE.TRIANGLE, 
                base: 16, altitude: 32, 
                color: 0xffc0cb            
            },
            prefab: {
                type: PsyanimArriveAgentPrefab,
            },
            components: [
                {
                    type: PsyanimArriveAgent,
                    params: {
                        target: {
                            entityName: 'mouseFollowTarget'
                        }
                    }
                }
            ]
        }
    ]
}
