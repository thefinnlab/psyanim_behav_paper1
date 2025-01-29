import { 
    PsyanimConstants,
    PsyanimMouseFollowTarget,
    PsyanimVehicle,
    PsyanimArriveBehavior,
    PsyanimArriveAgent,
} from 'psyanim2';

export default {
    key: 'MyArriveScene',
    wrapScreenBoundary: false,
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
            components: [
                {
                    type: PsyanimVehicle,
                },
                {
                    type: PsyanimArriveBehavior,
                    params: {
                        maxSpeed: 8,
                        innerDecelerationRadius: 25,
                        outerDecelerationRadius: 140
                    }
                },
                {
                    type: PsyanimArriveAgent,
                    params: {
                        arriveBehavior: {
                            entityName: 'agent1',
                            componentType: PsyanimArriveBehavior
                        },
                        vehicle: {
                            entityName: 'agent1',
                            componentType: PsyanimVehicle
                        },
                        target: {
                            entityName: 'mouseFollowTarget'
                        }
                    }
                }
            ]
        }
    ],
}