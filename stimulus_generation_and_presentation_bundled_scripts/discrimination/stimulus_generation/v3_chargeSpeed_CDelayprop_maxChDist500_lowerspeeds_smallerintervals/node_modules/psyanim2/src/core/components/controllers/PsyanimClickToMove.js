import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

import PsyanimConstants from '../../PsyanimConstants.js';

export default class PsyanimClickToMove extends PsyanimComponent {

    grid;

    pathfinder;

    arriveAgent;

    constructor(entity) {

        super(entity);

        this._pathFollowingTarget = this.scene.addEntity(this.entity.name + '_pathFollowingTarget', 0, 0, {
            shapeType: PsyanimConstants.SHAPE_TYPE.CIRCLE,
            radius: 4,
            color: 0xFFA500
        },
        {
            collisionFilter: PsyanimConstants.DEFAULT_VISUAL_ONLY_COLLISION_FILTER
        });

        this._pathFollowingTarget.position = this.entity.position;

        this._pathFollowingTarget.visible = false;

        this.scene.input.on('pointerup', this._handlePointerUp, this);

        this._state = PsyanimClickToMove.STATE.IDLE;
    }

    get state() {

        return this._state;
    }

    cancelMovementRequest() {

        this._state = PsyanimClickToMove.STATE.IDLE;

        this.arriveAgent.enabled = false;
    }

    _handlePointerUp(pointer) {

        if (pointer.leftButtonReleased())
        {
            this.arriveAgent.enabled = true;

            let newDestination = new Phaser.Math.Vector2(pointer.x, pointer.y);

            if (this.grid.isWorldPointInWalkableRegion(newDestination))
            {
                this.pathfinder.setDestination(newDestination);
            }

            this._state = PsyanimClickToMove.STATE.TRAVELING;
        }
    }

    _computePathfollowingTargetLocation() {

        let targetPositionOffset = 50;
        let targetParameterOffset = targetPositionOffset / this.pathfinder.currentPath.getTotalLength();

        let parameter = this.pathfinder.currentPath
            .getParameter(this.entity.position);

        let targetParameter = parameter + targetParameterOffset;

        let targetPosition = this.pathfinder.currentPath
            .getPosition(targetParameter);

        this._pathFollowingTarget.x = targetPosition.x;
        this._pathFollowingTarget.y = targetPosition.y;
    }

    update(t, dt) {

        super.update(t, dt);

        if (this._state == PsyanimClickToMove.STATE.TRAVELING)
        {
            if (this.arriveAgent)
            {
                this.arriveAgent.target = this._pathFollowingTarget;
            }
    
            this._computePathfollowingTargetLocation();
        }
    }
}

PsyanimClickToMove.STATE = {
    IDLE: 0x0000,
    TRAVELING: 0x0001
};