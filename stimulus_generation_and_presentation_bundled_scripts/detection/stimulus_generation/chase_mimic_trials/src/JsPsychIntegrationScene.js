import 
{
    PsyanimConstants,
    PsyanimWanderAgentPrefab
} from 'psyanim2';

export default {

    key: 'Wander Test',
    entities: [
        {
            name: 'agent',
            instances: 20,
            initialPosition: 'random',
            shapeParams: {
                shapeType: PsyanimConstants.SHAPE_TYPE.TRIANGLE, 
                base: 16, altitude: 32, 
                color: 0xffc0cb            
            },
            prefab: {
                type: PsyanimWanderAgentPrefab,
                params: {
                    debug: false,
                    radius: 150,
                    maxSpeed: 3,
                    maxAcceleration: 0.03,
                    maxAngleChangePerFrame: 10
                }
            }
        }
    ]
};