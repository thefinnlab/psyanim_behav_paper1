import PsyanimConstants from '../PsyanimConstants.js';
import PsyanimEntity from '../PsyanimEntity.js';

export default class PsyanimScreenBoundary {

    constructor(scene, x = 400, y = 300, width = 800, height = 600, wrap = true) {

        this.scene = scene;

        this.body = scene.matter.add.rectangle(x, y, width, height, {
            label: 'screen_boundary',
            isStatic: true,
            isSensor: true,
            collisionFilter: PsyanimConstants.DEFAULT_SCREEN_BOUNDARY_COLLISION_FILTER,
            onCollideEndCallback: (pair) => this._handleCollisionEnd(pair)
        });

        this.lowerYBound = y + height / 2;
        this.upperYBound = y - height / 2;
        this.leftXBound = x - width / 2;
        this.rightXBound = x + width/2;

        // setup collision boundaries
        const thickness = 400;

        this.topBoundary = scene.addEntity('topBoundary', width / 2, -thickness / 2, {
            shapeType: PsyanimConstants.SHAPE_TYPE.RECTANGLE, 
            width: 1.5 * width, height: 400,
            visible: false
        },
        { 
            isStatic: true, 
            isSleeping: true,
            collisionFilter: PsyanimConstants.DEFAULT_SCREEN_BOUNDARY_COLLISION_FILTER,
        });

        this.bottomBoundary = scene.addEntity('bottomBoundary', width / 2, height + thickness / 2, {
            shapeType: PsyanimConstants.SHAPE_TYPE.RECTANGLE, 
            width: 1.5 * width, height: thickness,
            visible: false
        }, 
        { 
            isStatic: true, 
            isSleeping: true,
            collisionFilter: PsyanimConstants.DEFAULT_SCREEN_BOUNDARY_COLLISION_FILTER,
        });

        this.leftBoundary = scene.addEntity('leftBoundary', -thickness / 2, height / 2, {
            shapeType: PsyanimConstants.SHAPE_TYPE.RECTANGLE, 
            width: thickness, height: 1.5 * height,
            visible: false
        }, 
        { 
            isStatic: true, 
            isSleeping: true,
            collisionFilter: PsyanimConstants.DEFAULT_SCREEN_BOUNDARY_COLLISION_FILTER,
        });

        this.rightBoundary = scene.addEntity('rightBoundary', width + thickness / 2, height / 2, {
            shapeType: PsyanimConstants.SHAPE_TYPE.RECTANGLE, 
            width: thickness, height: 1.5 * height,
            visible: false
        }, 
        { 
            isStatic: true, 
            isSleeping: true,
            collisionFilter: PsyanimConstants.DEFAULT_SCREEN_BOUNDARY_COLLISION_FILTER,
        });

        this.width = width;
        this.height = height;

        this.wrap = wrap;
    }

    isPointInBounds(point) {

        return point.x > 0 && point.x < this.width &&
                point.y > 0 && point.y < this.height;
    }

    /**
     * @return {boolean} - if true, characters crossing the screen boundary will be wrapped around to the other side, as if the world is spherical.
     */
    get wrap() {

        return this._wrap;
    }

    /**
     * @param {boolean} - if true, characters crossing the screen boundary will be wrapped around to the other side, as if the world is spherical.
     */
    set wrap(value) {

        this._wrap = value;

        this.topBoundary.body.isSensor = this._wrap;
        this.bottomBoundary.body.isSensor = this._wrap;
        this.leftBoundary.body.isSensor = this._wrap;
        this.rightBoundary.body.isSensor = this._wrap;
    }

    /**
     * @return {PsyanimEntity[]} - an array of entities this screen boundary is composed of: top, bottom, left, and right boundaries, respectively
     */
    get boundaries() {

        return [
            this.topBoundary, this.bottomBoundary,
            this.leftBoundary, this.rightBoundary
        ];
    }

    _handleCollisionEnd(pair) {

        let body = null;

        if (pair.bodyA !== this.body) {
            body = pair.bodyA;
        }
        else {
            body = pair.bodyB;
        }

        if (this._wrap)
        {
            this._wrapEntity(body);
        }
    }

    _wrapEntity(body)
    {
        if (body.position.y > this.lowerYBound)
        {
            // body passed the lower-Y boundary
            let delta = body.position.y - this.lowerYBound;

            this.scene.matter.body.setPosition(body, {x: body.position.x, y: this.upperYBound + delta});
        }

        if (body.position.y < this.upperYBound)
        {
            // body passed the upper-Y boundary
            let delta = this.upperYBound - body.position.y;

            this.scene.matter.body.setPosition(body, {x: body.position.x, y: this.lowerYBound - delta});
        }

        if (body.position.x > this.rightXBound)
        {
            // body passed the right-X boundary
            let delta = body.position.x - this.rightXBound;

            this.scene.matter.body.setPosition(body, {x: this.leftXBound + delta, y: body.position.y});
        }

        if (body.position.x < this.leftXBound)
        {
            // body passed the left-X boundary
            let delta = this.leftXBound - body.position.x;

            this.scene.matter.body.setPosition(body, {x: this.rightXBound - delta, y: body.position.y});
        }
    }
}