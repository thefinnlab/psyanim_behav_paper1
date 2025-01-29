import PsyanimFSM from '../PsyanimFSM.js';

import PsyanimVehicle from '../../steering/PsyanimVehicle.js';
import PsyanimSeekBehavior from '../../steering/PsyanimSeekBehavior.js';
import PsyanimWanderBehavior from '../../steering/PsyanimWanderBehavior.js';
import PsyanimFleeBehavior from '../../steering/PsyanimFleeBehavior.js';


import PsyanimPreyWanderState from './PsyanimPreyWanderState.js';
import PsyanimPreyFleeState from './PsyanimPreyFleeState.js';
import PsyanimPreyWallAvoidanceState from './PsyanimPreyWallAvoidanceState.js';

export default class PsyanimPreyFSM extends PsyanimFSM {

    /***********************************************************************************************/
    /************************************ Prey FSM Parameters **************************************/
    /***********************************************************************************************/

    /**
     *  Agent which this prey will watch and avoid.
     *  @type {PsyanimEntity}
     */
    target;

    /**
     *  Maximum number of degrees which this agent will offset it's flee direction.
     *  @type {Number}
     */
    subtlety;

    /**
     *  Determines how often the agent will recompute it's flee direction based on the `subtlety` parameter.
     *  @type {Number}
     */    
    subtletyLag;

    /***********************************************************************************************/
    /********************************** Wander State Parameters ************************************/
    /***********************************************************************************************/

    /**
     *  Maximum speed at which the agent will wander.
     *  @type {Number}
     */
    maxWanderSpeed;

    /**
     *  Maximum acceleration the agent can attain during wander.
     *  @type {Number}
     */
    maxWanderAcceleration;
    
    /**
     *  Radius of the wander circle.
     *  @type {Number}
     */
    wanderRadius;
    
    /**
     *  Distance the wander circle is offset from the agent's position.
     *  @type {Number}
     */
    wanderOffset;
    
    /**
     *  Maximum number of degrees the wander target can move around the wander circle per frame.
     *  @type {Number}
     */
    maxWanderAngleChangePerFrame;

    /***********************************************************************************************/
    /*********************************** Flee State Parameters *************************************/
    /***********************************************************************************************/

    /**
     *  Min. distance, in 'px', to target in which this agent will flee to maintain.
     *  @type {Number}
     */
    panicDistance;

    /**
     *  Max. speed agent can travel at when fleeing from target agent
     *  @type {Number}
     */
    maxFleeSpeed;

    /**
     *  Max. rate at which agent can accelerate when fleeing from target agent
     *  @type {Number}
     */
    maxFleeAcceleration;

    /**
     *  Minimum distance from all screen boundaries required for this agent to remain in 'Flee' state.
     * 
     *  If distance to any screen boundary is less than this distance, agent will transition to wall avoidance.
     *  @type {Number}
     */
    minimumWallSeparation;

    /***********************************************************************************************/
    /******************************** Wall Avoidance State Parameters ******************************/
    /***********************************************************************************************/

    /**
     *  List of points in the world where agent can seek away from wall before returning to a wander state.
     * 
     *  @type {Object[]}
     *  @property {Number} x
     *  @property {Number} y
     */
    seekTargetLocations;

    /**
     *  Distance from current seek target before agent is considered to have reached it 
     *  and transitions back to wander state.
     * 
     *  @type {Number}
     */
    seekTargetStoppingDistance;

    constructor(entity) {

        super(entity);

        // prey FSM params
        this.subtlety = 30;
        this.subtletyLag = 500;

        // wander state params
        this.maxWanderSpeed = 3;
        this.maxWanderAcceleration = 0.2;
        this.wanderRadius = 50;
        this.wanderOffset = 250;
        this.maxWanderAngleChangePerFrame = 20;

        // flee state params
        this.panicDistance = 250;
        this.maxFleeSpeed = 3;
        this.maxFleeAcceleration = 0.2;
        this.minimumWallSeparation = 50;

        // wall avoidance params
        this.seekTargetStoppingDistance = 50;

        this.seekTargetLocations = [
            { x: 400, y: 450 }, // bottom middle
            { x: 150, y: 450 }, // bottom left
            { x: 650, y: 450 }, // bottom right
            { x: 400, y: 150 }, // top middle
            { x: 150, y: 150 }, // top left
            { x: 650, y: 150 }, // top right
            { x: 150, y: 300 }, // left middle
            { x: 650, y: 300 }, // right middle
        ];

        // attach behaviors for this FSM
        this._vehicle = this.entity.addComponent(PsyanimVehicle);
        this._seekBehavior = this.entity.addComponent(PsyanimSeekBehavior);
        this._wanderBehavior = this.entity.addComponent(PsyanimWanderBehavior);
        this._fleeBehavior = this.entity.addComponent(PsyanimFleeBehavior);

        // setup fsm
        this._wanderState = this.addState(PsyanimPreyWanderState);
        this._fleeState = this.addState(PsyanimPreyFleeState);
        this._wallAvoidanceState = this.addState(PsyanimPreyWallAvoidanceState);

        this.initialState = this._wanderState;
    }

    onEnable() {

        super.onEnable();
    }

    onDisable() {

        super.onDisable();
    }

    afterCreate() {

        super.afterCreate();

        // wander state
        this._wanderState.target = this.target;

        // flee state
        this._fleeState.target = this.target;
        this._fleeState.subtlety = this.subtlety;
        this._fleeState.subtletyLag = this.subtletyLag;

        // wall avoidance state
        this._wallAvoidanceState.target = this.target;
        this._wallAvoidanceState.subtlety = this.subtlety;
        this._wallAvoidanceState.subtletyLag = this.subtletyLag;

        // wander behavior
        this._wanderBehavior.seekBehavior = this._seekBehavior;
        this._wanderBehavior.maxSeekSpeed = this.maxWanderSpeed;
        this._wanderBehavior.maxSeekAcceleration = this.maxWanderAcceleration;
        this._wanderBehavior.radius = this.wanderRadius;
        this._wanderBehavior.offset = this.wanderOffset;
        this._wanderBehavior.maxWanderAngleChangePerFrame = this.maxWanderAngleChangePerFrame;

        // flee behavior
        this._fleeBehavior.maxSpeed = this.maxFleeSpeed;
        this._fleeBehavior.maxAcceleration = this.maxFleeAcceleration;
        this._fleeBehavior.panicDistance = this.panicDistance;
    }

    beforeShutdown() {

        super.beforeShutdown();
    }

    update(t, dt) {

        super.update(t, dt);
    }
}