import { PsyanimConstants } from 'psyanim2';

import MyFirstMovementComponent from './MyFirstMovementComponent';

export default {
    key: 'MyFirstScene',
    wrapScreenBoundary: false,
    entities: [
        {
            name: 'agent1',
            initialPosition: { x: 400, y: 300 },
            shapeParams: {
                shapeType: PsyanimConstants.SHAPE_TYPE.CIRCLE, 
                radius: 10, 
                color: 0xff0000
            },
            components: [
                { type: MyFirstMovementComponent }
            ]
        }
    ]
}