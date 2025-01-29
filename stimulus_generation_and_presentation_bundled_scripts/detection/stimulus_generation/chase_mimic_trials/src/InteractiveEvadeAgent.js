import 
{ 
    PsyanimConstants,
    PsyanimPlayerController,
    PsyanimEvadeAgentPrefab,
    PsyanimEvadeAgent,
    PsyanimSensor,
    PsyanimJsPsychPlayerContactListener
    
} from 'psyanim2';

export default {

    key: 'InteractiveEvadeAgent',
    wrapScreenBoundary: false,
    entities: [
        {
            name: 'player',
            initialPosition: { x: 400, y: 300 },
            shapeParams: {
                shapeType: PsyanimConstants.SHAPE_TYPE.CIRCLE,
                radius: 12,
                color: 0x0000ff
            },
            components: [
                { 
                    type: PsyanimPlayerController,
                    params: {
                        speed: 12
                    }
                },
                {
                    type: PsyanimSensor,
                    params: {
                        bodyShapeParams: {
                            radius: 18,
                            shapeType: PsyanimConstants.SHAPE_TYPE.CIRCLE
                        }
                    }
                },
                {
                    type: PsyanimJsPsychPlayerContactListener,
                    params: {
                        sensor: {
                            entityName: 'player',
                            componentType: PsyanimSensor
                        },
                        targetEntityNames: [ 
                            "agent1",
                            "northWall",
                            "southWall"
                        ]
                    }
                }
            ]
        },
        
        {
            name: 'northWall',
            initialPosition: { x: 400, y: 15 },
            shapeParams: {
                shapeType: PsyanimConstants.SHAPE_TYPE.RECTANGLE,
                width: 720, height: 30, color: 0xff0000
            },
            matterOptions: {
                isStatic: true,
            }
        },
        {
            name: 'southWall',
            initialPosition: { x: 400, y: 585 },
            shapeParams: {
                shapeType: PsyanimConstants.SHAPE_TYPE.RECTANGLE,
                width: 720, height: 30, color: 0xff0000
            },
            matterOptions: {
                isStatic: true,
            }
        },
        {
            name: 'westWall',
            initialPosition: { x: 20, y: 300 },
            shapeParams: {
                shapeType: PsyanimConstants.SHAPE_TYPE.RECTANGLE,
                width: 40, height: 600, color: 0x00ff00
            },
            matterOptions: {
                isStatic: true,
            }
        },
        {
            name: 'eastWall',
            initialPosition: { x: 780, y: 300 },
            shapeParams: {
                shapeType: PsyanimConstants.SHAPE_TYPE.RECTANGLE,
                width: 40, height: 600, color: 0x00ff00
            },
            matterOptions: {
                isStatic: true,
            }
        },
        {
            name: 'agent1',
            initialPosition: { x: 600, y: 450 },
            shapeParams: {
                shapeType: PsyanimConstants.SHAPE_TYPE.TRIANGLE, 
                base: 16, altitude: 32, color: 0xff0000
            },
            prefab: { 
                type: PsyanimEvadeAgentPrefab,
                params: {
                    maxSpeed: 3,
                }
            },
            components: [
                {
                    type: PsyanimEvadeAgent,
                    params: {
                        target: {
                            entityName: 'player'
                        }
                    }
                }
            ]
        }
    ]
};